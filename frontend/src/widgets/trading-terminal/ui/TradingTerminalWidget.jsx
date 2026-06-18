import { useEffect, useMemo, useRef, useState } from "react";
import { getCandles, getFeaturePreview, getPrediction } from "../../../shared/api";

const SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"];
const MODEL_OPTIONS = [
  { value: "xgboost",  label: "XGBoost" },
  { value: "lstm",     label: "LSTM" },
  { value: "arima",    label: "ARIMA" },
  { value: "ensemble", label: "Ensemble" },
];
const ROLE_OPTIONS = [
  { value: "broker", label: "Broker" },
  { value: "admin", label: "Admin" },
];
const MIN_FORECAST_DAYS = 1;
const MAX_FORECAST_DAYS = 30;
const LIVE_INTERVAL_OPTIONS = [5, 10, 15, 30, 60];
const HOLD_MOVE_THRESHOLD_PCT = 0.35;
const STRONG_MOVE_THRESHOLD_PCT = 1.5;
const LOW_VOLATILITY_THRESHOLD_PCT = 1.2;
const HIGH_VOLATILITY_THRESHOLD_PCT = 2.8;
const MIN_STOP_DISTANCE_PCT = 0.7;
const MAX_STOP_DISTANCE_PCT = 4;
const BASE_FIELD_HINTS = {
  lastClose:
    "Последняя цена закрытия выбранного актива в последней свече. Используйте как базовую точку для сравнения с прогнозом.",
  forecastHorizon:
    "Горизонт прогноза в днях. Чем дальше горизонт, тем выше неопределенность результата.",
  trainRows:
    "Количество строк, на которых обучалась модель. Больше данных обычно дает более стабильный прогноз.",
  mae: "Средняя абсолютная ошибка на валидации (в USD). Ниже значение — лучше качество модели.",
  rmse: "Корень из средней квадратичной ошибки (в USD). Сильнее штрафует большие промахи, чем MAE.",
  mape:
    "Средняя абсолютная процентная ошибка. Показывает относительную ошибку прогноза в процентах от фактической цены.",
  mse: "Средняя квадратичная ошибка. Полезна для сравнения качества моделей при одинаковом масштабе данных.",
  modelId:
    "Уникальный идентификатор сохраненной модели. По нему видно, какая именно версия модели сейчас используется.",
  modelType:
    "Тип модели для прогноза. Ensemble усредняет XGBoost, ARIMA и LSTM. Одиночный режим использует только выбранную модель. Если модель не обучена — запустится авто-обучение.",
  lossFunction:
    "Функция потерь при обучении. Она определяет, как модель штрафуется за ошибки прогноза.",
  tuningScoring:
    "Метрика, по которой подбирались гиперпараметры. mae — ошибка в долларах, mape — относительная ошибка в процентах.",
  trainedAt:
    "Время последнего обучения модели. Если модель давно не обучалась, прогноз может быть менее актуальным.",
  trainingDataFreshness:
    "Свежесть данных, использованных для обучения. Fresh обычно означает более релевантную модель.",
  trainingMaxLag:
    "Максимальная задержка обучающих данных относительно текущего времени. Меньше задержка — лучше.",
  requestDataFreshness:
    "Свежесть входных данных текущего запроса на прогноз. Delayed может ухудшать точность.",
  requestMaxLag:
    "Максимальная задержка данных в текущем запросе. Большой lag означает, что прогноз строится на более старых свечах.",
};

function formatPrice(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatDate(value) {
  return new Date(value).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
  });
}

function formatDateTime(value) {
  if (!value) {
    return "—";
  }
  return new Date(value).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatLagMinutes(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return `${Number(value).toFixed(1)} min`;
}

function formatPercent(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return `${Number(value).toFixed(2)}%`;
}

function formatSignedPercent(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  const numericValue = Number(value);
  return `${numericValue >= 0 ? "+" : ""}${numericValue.toFixed(2)}%`;
}

function formatMetricValue(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatFeatureValue(feature, value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  if (feature?.scale_hint === "binary") {
    return Number(value) >= 0.5 ? "1" : "0";
  }
  if (feature?.scale_hint === "percent") {
    return formatPercent(Number(value) * 100);
  }
  if (feature?.scale_hint === "price") {
    return formatPrice(Number(value));
  }
  if (feature?.scale_hint === "volume") {
    return Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 4,
  });
}

function FeatureSparkline({ history, scaleHint, category, detailed = false }) {
  const seriesPoints = Array.isArray(history)
    ? history
      .map((item) => ({
        value: Number(item?.value),
        timestamp: item?.timestamp ?? null,
      }))
      .filter((item) => Number.isFinite(item.value))
    : [];
  const points = seriesPoints.map((item) => item.value);
  if (!points.length) {
    return <div className="feature-sparkline-empty">No data</div>;
  }
  const width = detailed ? 760 : 170;
  const height = detailed ? 220 : 48;
  const axis = detailed
    ? { top: 18, right: 16, bottom: 30, left: 58 }
    : { top: 6, right: 6, bottom: 12, left: 24 };
  const drawWidth = Math.max(width - axis.left - axis.right, 1);
  const drawHeight = Math.max(height - axis.top - axis.bottom, 1);
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(max - min, 1e-9);
  const isBarChart = category === "volume" || category === "symbol";
  const isOscillator = category === "oscillator";
  const isLineChart = !isBarChart;
  const toX = (idx) => axis.left + (idx / Math.max(points.length - 1, 1)) * drawWidth;
  const toY = (value) => axis.top + ((max - value) / span) * drawHeight;
  const polyline = points.map((value, idx) => `${toX(idx)},${toY(value)}`).join(" ");
  const latest = points[points.length - 1];
  const earliest = points[0];
  const trendClass = latest >= earliest ? "up" : "down";
  const showZeroLine = isLineChart && (scaleHint === "percent" || scaleHint === "number" || isOscillator);
  const zeroY = min <= 0 && max >= 0 ? toY(0) : null;
  const barSlotWidth = drawWidth / Math.max(points.length, 1);
  const barWidth = Math.max(barSlotWidth - (detailed ? 1.2 : 0.8), detailed ? 2 : 1);
  const barBottom = axis.top + drawHeight;
  const oscillatorUpperY = axis.top + drawHeight * 0.2;
  const oscillatorLowerY = axis.top + drawHeight * 0.8;
  const oscillatorZoneY = axis.top + drawHeight * 0.2;
  const oscillatorZoneHeight = drawHeight * 0.6;
  const yTickCount = detailed ? 5 : 3;
  const yTicks = Array.from({ length: yTickCount }, (_, tickIndex) => {
    const ratio = tickIndex / Math.max(yTickCount - 1, 1);
    const value = max - ratio * span;
    return {
      value,
      y: toY(value),
    };
  });
  const xTickCount = detailed ? Math.min(6, points.length) : Math.min(3, points.length);
  const xTickIndexes = Array.from({ length: xTickCount }, (_, tickIndex) =>
    Math.round((tickIndex * Math.max(points.length - 1, 0)) / Math.max(xTickCount - 1, 1))
  ).filter((value, index, items) => items.indexOf(value) === index);
  const showTickLabels = detailed;
  const showAxisTitles = detailed;
  const hasTimeAxis = seriesPoints.some((item) => Boolean(item.timestamp));

  function formatAxisTimestamp(timestamp, index) {
    if (!timestamp) {
      return `${index + 1}`;
    }
    return new Date(timestamp).toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
    });
  }

  function formatAxisValue(value) {
    if (!Number.isFinite(value)) {
      return "—";
    }
    if (scaleHint === "binary") {
      return value >= 0.5 ? "1" : "0";
    }
    if (scaleHint === "percent") {
      return `${(value * 100).toFixed(detailed ? 1 : 0)}%`;
    }
    if (scaleHint === "price") {
      return Number(value).toLocaleString("en-US", {
        maximumFractionDigits: detailed ? 2 : 0,
      });
    }
    if (scaleHint === "volume") {
      return Number(value).toLocaleString("en-US", {
        notation: "compact",
        maximumFractionDigits: 1,
      });
    }
    return Number(value).toLocaleString("en-US", {
      maximumFractionDigits: detailed ? 3 : 1,
    });
  }

  const yAxisTitle =
    scaleHint === "price"
      ? "Price"
      : scaleHint === "percent"
        ? "Percent"
        : scaleHint === "volume"
          ? "Volume"
          : scaleHint === "binary"
            ? "State"
            : "Value";
  const xAxisTitle = hasTimeAxis ? "Time" : "Samples";

  return (
    <svg
      className={`feature-sparkline feature-sparkline-${trendClass} ${detailed ? "feature-sparkline-detailed" : ""}`}
      viewBox={`0 0 ${width} ${height}`}
    >
      <rect x={0} y={0} width={width} height={height} rx={10} className="feature-sparkline-bg" />
      {yTicks.map((tick, idx) => (
        <g key={`feature-y-tick-${idx}`}>
          <line
            x1={axis.left}
            y1={tick.y}
            x2={width - axis.right}
            y2={tick.y}
            className="feature-sparkline-grid"
          />
          {showTickLabels && (
            <text
              x={axis.left - 6}
              y={tick.y + 3}
              textAnchor="end"
              className="feature-sparkline-tick"
            >
              {formatAxisValue(tick.value)}
            </text>
          )}
        </g>
      ))}
      <line
        x1={axis.left}
        y1={axis.top}
        x2={axis.left}
        y2={axis.top + drawHeight}
        className="feature-sparkline-axis"
      />
      <line
        x1={axis.left}
        y1={axis.top + drawHeight}
        x2={width - axis.right}
        y2={axis.top + drawHeight}
        className="feature-sparkline-axis"
      />
      {xTickIndexes.map((index) => {
        const x = toX(index);
        return (
          <g key={`feature-x-tick-${index}`}>
            <line
              x1={x}
              y1={axis.top + drawHeight}
              x2={x}
              y2={axis.top + drawHeight + (showTickLabels ? 6 : 4)}
              className="feature-sparkline-axis"
            />
            {showTickLabels && (
              <text
                x={x}
                y={height - 6}
                textAnchor="middle"
                className="feature-sparkline-tick"
              >
                {formatAxisTimestamp(seriesPoints[index]?.timestamp, index)}
              </text>
            )}
          </g>
        );
      })}
      {showAxisTitles && (
        <>
          <text x={axis.left} y={axis.top - 7} className="feature-sparkline-label">
            {yAxisTitle}
          </text>
          <text
            x={axis.left + drawWidth / 2}
            y={height - 2}
            textAnchor="middle"
            className="feature-sparkline-label"
          >
            {xAxisTitle}
          </text>
        </>
      )}
      {isOscillator && (
        <>
          <rect
            x={axis.left}
            y={oscillatorZoneY}
            width={drawWidth}
            height={oscillatorZoneHeight}
            className="feature-sparkline-osc-zone"
          />
          <line
            x1={axis.left}
            y1={oscillatorUpperY}
            x2={width - axis.right}
            y2={oscillatorUpperY}
            className="feature-sparkline-osc-line"
          />
          <line
            x1={axis.left}
            y1={oscillatorLowerY}
            x2={width - axis.right}
            y2={oscillatorLowerY}
            className="feature-sparkline-osc-line"
          />
        </>
      )}
      {showZeroLine && zeroY !== null && (
        <line
          x1={axis.left}
          y1={zeroY}
          x2={width - axis.right}
          y2={zeroY}
          className="feature-sparkline-zero"
        />
      )}
      {isBarChart ? (
        <g>
          {points.map((value, idx) => {
            const x = axis.left + idx * barSlotWidth + (barSlotWidth - barWidth) / 2;
            const y = toY(value);
            const h = Math.max(barBottom - y, 1.5);
            const directionClass =
              idx === 0 || value >= points[idx - 1] ? "feature-sparkline-bar-up" : "feature-sparkline-bar-down";
            return (
              <rect
                key={`bar-${idx}`}
                x={x}
                y={y}
                width={barWidth}
                height={h}
                rx={barWidth > 2 ? 1.5 : 0.5}
                className={`feature-sparkline-bar ${directionClass}`}
              />
            );
          })}
        </g>
      ) : (
        <polyline points={polyline} fill="none" className="feature-sparkline-line" />
      )}
    </svg>
  );
}

function StatLabel({ text, hint }) {
  return (
    <span className="stat-label">
      <span>{text}</span>
      <abbr className="stat-hint" title={hint}>
        ⓘ
      </abbr>
    </span>
  );
}
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatCompactNumber(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatAxisPrice(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }
  const absValue = Math.abs(Number(value));
  const fractionDigits = absValue >= 1000 ? 0 : absValue >= 50 ? 1 : 2;
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

function formatChartTooltipDate(value) {
  if (!value) {
    return "—";
  }
  return new Date(value).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function LegacyCandlestickChartUnused({
  candles,
  forecastPath,
  forecastSpreadPath = [],
  selectedForecastDay,
  onSelectForecastDay,
  showTechnical = true,
  levelOverlays = [],
}) {
  const chartContainerRef = useRef(null);
  const chartSvgRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(0);
  const chartHeight = 430;
  const chartMargin = { left: 64, right: 140, top: 12, bottom: 30 };
  const innerChartHeight = chartHeight - chartMargin.top - chartMargin.bottom;
  const volumeChartHeight = 96;
  const mainChartHeight = Math.max(innerChartHeight - volumeChartHeight, 180);
  const ratio = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const axisPriceFormat = useMemo(() => d3Format(",.0f"), []);
  const priceDisplayFormat = useMemo(() => d3Format(",.2f"), []);
  const volumeDisplayFormat = useMemo(() => d3Format(".3s"), []);
  const dateDisplayFormat = useMemo(() => timeFormat("%d %b"), []);

  useEffect(() => {
    const node = chartContainerRef.current;
    if (!node) {
      return undefined;
    }
    const observedElement = node.parentElement || node;
    const updateWidth = () => {
      const observedWidth = observedElement.getBoundingClientRect().width || 0;
      const ownWidth = node.getBoundingClientRect().width || 0;
      const next = Math.floor(observedWidth || ownWidth || 0);
      if (next > 0 && Number.isFinite(next)) {
        setChartWidth(Math.max(320, next - 2));
      }
    };
    updateWidth();
    const rafId = requestAnimationFrame(updateWidth);
    let observer;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => updateWidth());
      observer.observe(observedElement);
    }
    window.addEventListener("resize", updateWidth);
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", updateWidth);
      observer?.disconnect();
    };
  }, []);

  const overlayItems = useMemo(() => {
    const toneToColor = {
      up: "#427b58",
      down: "#9d0006",
      neutral: "#665c54",
    };
    const toneToDash = {
      up: "ShortDash",
      down: "ShortDash",
      neutral: "Dot",
    };
    return Array.isArray(levelOverlays)
      ? levelOverlays
        .map((item, index) => {
          const value = Number(item?.value);
          if (!Number.isFinite(value)) {
            return null;
          }
          const tone = item?.tone || "neutral";
          return {
            key: item?.key || `overlay-${index}`,
            label: item?.label || "Level",
            tone,
            value,
            color: toneToColor[tone] || toneToColor.neutral,
            dash: toneToDash[tone] || "ShortDash",
          };
        })
        .filter(Boolean)
      : [];
  }, [levelOverlays]);

  const forecastSpreadByDay = useMemo(() => {
    const spreadMap = new Map();
    forecastSpreadPath.forEach((item) => {
      const dayAhead = Number(item?.dayAhead);
      const lower = Number(item?.lower);
      const upper = Number(item?.upper);
      const spread = Number(item?.spread);
      const spreadPct = Number(item?.spreadPct);
      if (!Number.isFinite(dayAhead) || !Number.isFinite(lower) || !Number.isFinite(upper)) {
        return;
      }
      spreadMap.set(dayAhead, {
        lower: Math.min(lower, upper),
        upper: Math.max(lower, upper),
        spread: Number.isFinite(spread) ? Math.max(spread, 0) : null,
        spreadPct: Number.isFinite(spreadPct) ? Math.max(spreadPct, 0) : null,
      });
    });
    return spreadMap;
  }, [forecastSpreadPath]);

  const mergedData = useMemo(() => {
    const normalizedHistorical = candles
      .map((item, index) => {
        const close = Number(item?.close);
        if (!Number.isFinite(close)) {
          return null;
        }
        const openCandidate = Number(item?.open);
        const highCandidate = Number(item?.high);
        const lowCandidate = Number(item?.low);
        const volumeCandidate = Number(item?.volume);
        const dateCandidate = item?.timestamp ? new Date(item.timestamp) : null;
        const date =
          dateCandidate && Number.isFinite(dateCandidate.getTime())
            ? dateCandidate
            : new Date(Date.now() - (candles.length - index) * 24 * 60 * 60 * 1000);
        const open = Number.isFinite(openCandidate) ? openCandidate : close;
        const high = Number.isFinite(highCandidate)
          ? Math.max(highCandidate, open, close)
          : Math.max(open, close);
        const low = Number.isFinite(lowCandidate)
          ? Math.min(lowCandidate, open, close)
          : Math.min(open, close);
        return {
          date,
          open,
          high,
          low,
          close,
          volume: Number.isFinite(volumeCandidate) ? volumeCandidate : 0,
          isForecast: false,
          dayAhead: null,
          isSelected: false,
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.date.getTime() - b.date.getTime());
    if (!normalizedHistorical.length) {
      return [];
    }
    const recentRanges = normalizedHistorical
      .slice(-20)
      .map((item) => Math.max(item.high - item.low, 0))
      .filter((value) => Number.isFinite(value) && value > 0);
    const baseRange =
      recentRanges.length > 0
        ? recentRanges.reduce((sum, value) => sum + value, 0) / recentRanges.length
        : Math.max(normalizedHistorical[normalizedHistorical.length - 1].close * 0.01, 1);
    const wickPadding = baseRange * 0.35;
    const forecastRows = [];
    let previousClose = normalizedHistorical[normalizedHistorical.length - 1].close;
    let previousDate = normalizedHistorical[normalizedHistorical.length - 1].date;
    forecastPath.forEach((point, index) => {
      const predictedClose = Number(point?.predicted_price);
      if (!Number.isFinite(predictedClose)) {
        return;
      }
      const dayAhead = point?.day_ahead ?? index + 1;
      const spreadMeta = forecastSpreadByDay.get(Number(dayAhead));
      const parsedDate = point?.predict_for_at ? new Date(point.predict_for_at) : null;
      const fallbackDate = new Date(previousDate.getTime() + 24 * 60 * 60 * 1000);
      const nextDate =
        parsedDate && Number.isFinite(parsedDate.getTime()) && parsedDate > previousDate
          ? parsedDate
          : fallbackDate;
      const open = previousClose;
      const close = predictedClose;
      const high = Math.max(open, close) + wickPadding;
      const low = Math.max(0, Math.min(open, close) - wickPadding);
      forecastRows.push({
        date: nextDate,
        open,
        high,
        low,
        close,
        volume: 0,
        isForecast: true,
        dayAhead,
        isSelected: selectedForecastDay === dayAhead,
        uncertaintyLow: spreadMeta?.lower ?? null,
        uncertaintyHigh: spreadMeta?.upper ?? null,
        uncertaintySpread: spreadMeta?.spread ?? null,
        uncertaintySpreadPct: spreadMeta?.spreadPct ?? null,
      });
      previousClose = close;
      previousDate = nextDate;
    });
    return [...normalizedHistorical, ...forecastRows];
  }, [candles, forecastPath, forecastSpreadByDay, selectedForecastDay]);

  const chartData = useMemo(() => {
    if (!mergedData.length) {
      return null;
    }
    const scaleProvider = discontinuousTimeScaleProvider.inputDateAccessor((d) => d.date);
    return scaleProvider(mergedData);
  }, [mergedData]);

  if (!chartData || !chartData.data.length) {
    return <div className="chart-empty">No market data</div>;
  }

  const { data, xScale, xAccessor, displayXAccessor } = chartData;
  const visibleBars = Math.min(data.length, Math.max(72, Math.min(140, forecastPath.length + 64)));
  const startIndex = Math.max(0, data.length - visibleBars);
  const xExtents = [xAccessor(data[startIndex]), xAccessor(data[data.length - 1])];
  const yExtents = (d) => [d.high, d.low, ...overlayItems.map((item) => item.value)];
  const historicalOhlcAccessor = (d) =>
    d.isForecast
      ? { open: undefined, high: undefined, low: undefined, close: undefined }
      : { open: d.open, high: d.high, low: d.low, close: d.close };
  const forecastOhlcAccessor = (d) =>
    d.isForecast
      ? { open: d.open, high: d.high, low: d.low, close: d.close }
      : { open: undefined, high: undefined, low: undefined, close: undefined };
  const historicalVolumeAccessor = (d) => (d.isForecast ? undefined : d.volume);
  const seriesName = `market-${data[0]?.date?.getTime?.() ?? "na"}-${data[data.length - 1]?.date?.getTime?.() ?? "na"}-${data.length}`;
  const effectiveChartWidth = chartWidth > 0 ? chartWidth : 980;

  return (
    <div className="chart-viewport stockchart-viewport" ref={chartContainerRef}>
      {showTechnical && (
        <div className="chart-viewport-toolbar chart-viewport-toolbar-stock">
          <div className="chart-viewport-meta">
            <span>Drag = pan</span>
            <span>Wheel = zoom</span>
            <span>Scroll = zoom range</span>
          </div>
        </div>
      )}
      <ChartCanvas
        height={chartHeight}
        width={effectiveChartWidth}
        ratio={ratio}
        margin={chartMargin}
        type="hybrid"
        seriesName={seriesName}
        data={data}
        xScale={xScale}
        xAccessor={xAccessor}
        displayXAccessor={displayXAccessor}
        xExtents={xExtents}
        mouseMoveEvent
        panEvent
        zoomEvent
      >
        <Chart id={1} height={mainChartHeight} yExtents={yExtents} padding={{ top: 16, bottom: 18 }}>
          <XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
          <YAxis
            axisAt="right"
            orient="right"
            ticks={6}
            tickFormat={axisPriceFormat}
            tickLabelFill="#7c6f64"
            tickStroke="#d5c4a1"
            stroke="#7c6f64"
          />
          <MouseCoordinateX
            at="bottom"
            orient="bottom"
            displayFormat={dateDisplayFormat}
          />
          <MouseCoordinateY
            at="right"
            orient="right"
            displayFormat={priceDisplayFormat}
          />
          <CandlestickSeries
            yAccessor={historicalOhlcAccessor}
            opacity={0.9}
          />
          <CandlestickSeries
            yAccessor={forecastOhlcAccessor}
            classNames={(d) => `forecast ${d.close >= d.open ? "up" : "down"} ${d.isSelected ? "selected" : ""}`}
            fill={(d) =>
              d.isSelected ? "rgba(69, 133, 136, 0.55)" : "rgba(69, 133, 136, 0.32)"
            }
            stroke="#3f6f7a"
            wickStroke="#3f6f7a"
            candleStrokeWidth={1}
            opacity={0.8}
          />
          {overlayItems.map((overlay) => (
            <LineSeries
              key={`line-${overlay.key}`}
              yAccessor={() => overlay.value}
              stroke={overlay.color}
              strokeWidth={1.2}
              strokeDasharray={overlay.dash}
            />
          ))}
          {overlayItems.map((overlay) => (
            <PriceCoordinate
              key={`price-${overlay.key}`}
              price={overlay.value}
              at="right"
              orient="right"
              lineStroke={overlay.color}
              stroke={overlay.color}
              fill={overlay.color}
              textFill="#fbf1c7"
              lineOpacity={0.75}
              strokeOpacity={0.95}
              rectWidth={122}
              rectHeight={16}
              fontSize={10}
              displayFormat={(value) => `${overlay.label} ${axisPriceFormat(value)}`}
            />
          ))}
          <CurrentCoordinate
            yAccessor={(d) => d.close}
            fill={(d) => (d.isForecast ? "#458588" : d.close >= d.open ? "#98971a" : "#cc241d")}
          />
          <EdgeIndicator
            itemType="last"
            orient="right"
            edgeAt="right"
            yAccessor={(d) => d.close}
            displayFormat={priceDisplayFormat}
            fill={(d) => (d.close >= d.open ? "#98971a" : "#cc241d")}
            textFill="#fbf1c7"
            lineStroke="#7c6f64"
          />
          <OHLCTooltip
            origin={[8, 12]}
            ohlcFormat={priceDisplayFormat}
            xDisplayFormat={timeFormat("%d %b %Y")}
            displayTexts={{
              d: "Date: ",
              o: " O: ",
              h: " H: ",
              l: " L: ",
              c: " C: ",
              v: " Vol: ",
              na: "n/a",
            }}
          />
        </Chart>
        <Chart
          id={2}
          height={volumeChartHeight}
          origin={(w, h) => [0, h - volumeChartHeight]}
          yExtents={(d) => historicalVolumeAccessor(d) || 0}
        >
          <XAxis
            axisAt="bottom"
            orient="bottom"
            ticks={6}
            tickFormat={dateDisplayFormat}
            tickLabelFill="#7c6f64"
            tickStroke="#d5c4a1"
            stroke="#7c6f64"
          />
          <YAxis
            axisAt="left"
            orient="left"
            ticks={3}
            tickFormat={volumeDisplayFormat}
            tickLabelFill="#7c6f64"
            tickStroke="#d5c4a1"
            stroke="#7c6f64"
          />
          <BarSeries
            yAccessor={historicalVolumeAccessor}
            fill={(d) =>
              d.close >= d.open ? "rgba(152, 151, 26, 0.45)" : "rgba(204, 36, 29, 0.45)"
            }
          />
        </Chart>
        <CrossHairCursor stroke="#7c6f64" />
      </ChartCanvas>
    </div>
  );
}


function CandlestickChart({
  candles,
  forecastPath,
  forecastSpreadPath = [],
  selectedForecastDay,
  onSelectForecastDay,
  showTechnical = true,
  levelOverlays = [],
}) {
  const chartContainerRef = useRef(null);
  const chartSvgRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(0);
  const [hoverState, setHoverState] = useState({ index: null, x: null });
  const [visibleRange, setVisibleRange] = useState(null);
  const chartHeight = 430;

  useEffect(() => {
    const node = chartContainerRef.current;
    if (!node) {
      return undefined;
    }
    const updateWidth = () => {
      const ownWidth = node.getBoundingClientRect().width || 0;
      const parentWidth = node.parentElement?.getBoundingClientRect().width || 0;
      const next = Math.floor(ownWidth || parentWidth || 0);
      if (next > 0 && Number.isFinite(next)) {
        setChartWidth((previousWidth) => (previousWidth === next ? previousWidth : next));
      }
    };
    updateWidth();
    const animationFrame = requestAnimationFrame(updateWidth);
    let observer = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => updateWidth());
      observer.observe(node);
      if (node.parentElement) {
        observer.observe(node.parentElement);
      }
    }
    window.addEventListener("resize", updateWidth);
    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener("resize", updateWidth);
      observer?.disconnect();
    };
  }, []);

  const overlayItems = useMemo(() => {
    const toneToColor = {
      up: "#427b58",
      down: "#9d0006",
      neutral: "#7c6f64",
    };
    const toneToDash = {
      up: "7 5",
      down: "7 5",
      neutral: "2 4",
    };
    return Array.isArray(levelOverlays)
      ? levelOverlays
      .map((item, index) => {
        const value = Number(item?.value);
        if (!Number.isFinite(value)) {
          return null;
        }
        const tone = item?.tone || "neutral";
        return {
          key: item?.key || `overlay-${index}`,
          label: item?.label || "Level",
          value,
          color: toneToColor[tone] || toneToColor.neutral,
          dash: toneToDash[tone] || "2 4",
        };
      })
      .filter(Boolean)
      : [];
  }, [levelOverlays]);

  const forecastSpreadByDay = useMemo(() => {
    const spreadMap = new Map();
    forecastSpreadPath.forEach((item) => {
      const dayAhead = Number(item?.dayAhead);
      const lower = Number(item?.lower);
      const upper = Number(item?.upper);
      const spread = Number(item?.spread);
      const spreadPct = Number(item?.spreadPct);
      if (!Number.isFinite(dayAhead) || !Number.isFinite(lower) || !Number.isFinite(upper)) {
        return;
      }
      spreadMap.set(dayAhead, {
        lower: Math.min(lower, upper),
        upper: Math.max(lower, upper),
        spread: Number.isFinite(spread) ? Math.max(spread, 0) : null,
        spreadPct: Number.isFinite(spreadPct) ? Math.max(spreadPct, 0) : null,
      });
    });
    return spreadMap;
  }, [forecastSpreadPath]);

  const mergedData = useMemo(() => {
    const historicalRows = candles
      .map((item, index) => {
        const close = Number(item?.close);
        if (!Number.isFinite(close)) {
          return null;
        }
        const openCandidate = Number(item?.open);
        const highCandidate = Number(item?.high);
        const lowCandidate = Number(item?.low);
        const volumeCandidate = Number(item?.volume);
        const parsedDate = item?.timestamp ? new Date(item.timestamp) : null;
        const date =
          parsedDate && Number.isFinite(parsedDate.getTime())
            ? parsedDate
            : new Date(Date.now() - (candles.length - index) * 24 * 60 * 60 * 1000);
        const open = Number.isFinite(openCandidate) ? openCandidate : close;
        const high = Number.isFinite(highCandidate)
          ? Math.max(highCandidate, open, close)
          : Math.max(open, close);
        const low = Number.isFinite(lowCandidate)
          ? Math.min(lowCandidate, open, close)
          : Math.min(open, close);
        return {
          id: `history-${index}`,
          date,
          open,
          high,
          low,
          close,
          volume: Number.isFinite(volumeCandidate) ? volumeCandidate : 0,
          isForecast: false,
          dayAhead: null,
          isSelected: false,
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.date.getTime() - b.date.getTime());

    if (!historicalRows.length) {
      return [];
    }

    const recentRanges = historicalRows
      .slice(-20)
      .map((item) => Math.max(item.high - item.low, 0))
      .filter((value) => Number.isFinite(value) && value > 0);

    const baseRange =
      recentRanges.length > 0
        ? recentRanges.reduce((sum, value) => sum + value, 0) / recentRanges.length
        : Math.max(historicalRows[historicalRows.length - 1].close * 0.01, 1);
    const wickPadding = baseRange * 0.35;
    const forecastRows = [];
    let previousClose = historicalRows[historicalRows.length - 1].close;
    let previousDate = historicalRows[historicalRows.length - 1].date;

    forecastPath.forEach((point, index) => {
      const predictedClose = Number(point?.predicted_price);
      if (!Number.isFinite(predictedClose)) {
        return;
      }
      const dayAhead = point?.day_ahead ?? index + 1;
      const spreadMeta = forecastSpreadByDay.get(Number(dayAhead));
      const parsedDate = point?.predict_for_at ? new Date(point.predict_for_at) : null;
      const fallbackDate = new Date(previousDate.getTime() + 24 * 60 * 60 * 1000);
      const nextDate =
        parsedDate && Number.isFinite(parsedDate.getTime()) && parsedDate > previousDate
          ? parsedDate
          : fallbackDate;
      const open = previousClose;
      const close = predictedClose;
      const high = Math.max(open, close) + wickPadding;
      const low = Math.max(0, Math.min(open, close) - wickPadding);
      forecastRows.push({
        id: `forecast-${dayAhead}-${index}`,
        date: nextDate,
        open,
        high,
        low,
        close,
        volume: 0,
        isForecast: true,
        dayAhead,
        isSelected: selectedForecastDay === dayAhead,
        uncertaintyLow: spreadMeta?.lower ?? null,
        uncertaintyHigh: spreadMeta?.upper ?? null,
        uncertaintySpread: spreadMeta?.spread ?? null,
        uncertaintySpreadPct: spreadMeta?.spreadPct ?? null,
      });
      previousClose = close;
      previousDate = nextDate;
    });

    return [...historicalRows, ...forecastRows];
  }, [candles, forecastPath, forecastSpreadByDay, selectedForecastDay]);

  useEffect(() => {
    if (!mergedData.length) {
      setVisibleRange(null);
      setHoverState({ index: null, x: null });
      return;
    }
    const defaultBars = Math.min(Math.max(forecastPath.length + 48, 56), mergedData.length);
    const fallbackStart = Math.max(0, mergedData.length - defaultBars);
    setVisibleRange((currentRange) => {
      if (!currentRange) {
        return {
          start: fallbackStart,
          end: mergedData.length - 1,
        };
      }
      const minBars = Math.min(16, mergedData.length);
      const nextStart = clamp(currentRange.start, 0, mergedData.length - 1);
      const nextEnd = clamp(currentRange.end, nextStart, mergedData.length - 1);
      const nextCount = clamp(nextEnd - nextStart + 1, minBars, mergedData.length);
      const normalizedStart = clamp(nextEnd - nextCount + 1, 0, mergedData.length - nextCount);
      return {
        start: normalizedStart,
        end: normalizedStart + nextCount - 1,
      };
    });
    setHoverState((currentHover) => {
      if (currentHover.index === null) {
        return currentHover;
      }
      return {
        index: clamp(currentHover.index, 0, mergedData.length - 1),
        x: null,
      };
    });
  }, [mergedData.length, forecastPath.length]);

  if (!mergedData.length) {
    return <div className="chart-empty">No market data</div>;
  }

  const minimumVisibleBars = Math.min(16, mergedData.length);
  const maximumVisibleBars = mergedData.length;
  const defaultVisibleBars = Math.min(Math.max(forecastPath.length + 48, 56), mergedData.length);
  const visibleStart = visibleRange?.start ?? Math.max(0, mergedData.length - defaultVisibleBars);
  const visibleEnd = visibleRange?.end ?? mergedData.length - 1;
  const visibleData = mergedData.slice(visibleStart, visibleEnd + 1);
  const visibleBars = Math.max(visibleData.length, 1);
  const effectiveChartWidth = chartWidth > 0 ? chartWidth : 980;
  const marginLeft = effectiveChartWidth < 520 ? 56 : 66;
  const marginRight = effectiveChartWidth < 760 ? 108 : 132;
  const marginTop = 18;
  const marginBottom = 30;
  const volumeHeight = 88;
  const panelGap = 14;
  const plotWidth = Math.max(effectiveChartWidth - marginLeft - marginRight, 120);
  const priceHeight = Math.max(chartHeight - marginTop - marginBottom - volumeHeight - panelGap, 180);
  const priceBottom = marginTop + priceHeight;
  const volumeTop = priceBottom + panelGap;
  const volumeBottom = volumeTop + volumeHeight;
  const candleSlotWidth = plotWidth / visibleBars;
  const candleBodyWidth = clamp(candleSlotWidth * 0.72, 3, 18);

  const lowValues = visibleData.map((item) => item.low);
  const highValues = visibleData.map((item) => item.high);
  const uncertaintyLowValues = visibleData
    .map((item) => Number(item?.uncertaintyLow))
    .filter((value) => Number.isFinite(value));
  const uncertaintyHighValues = visibleData
    .map((item) => Number(item?.uncertaintyHigh))
    .filter((value) => Number.isFinite(value));
  const overlayValues = overlayItems.map((item) => item.value);
  const rawPriceMin = Math.min(
    ...lowValues,
    ...(uncertaintyLowValues.length ? uncertaintyLowValues : [Number.POSITIVE_INFINITY]),
    ...(overlayValues.length ? overlayValues : [Number.POSITIVE_INFINITY])
  );
  const rawPriceMax = Math.max(
    ...highValues,
    ...(uncertaintyHighValues.length ? uncertaintyHighValues : [Number.NEGATIVE_INFINITY]),
    ...(overlayValues.length ? overlayValues : [Number.NEGATIVE_INFINITY])
  );
  const fallbackPrice = visibleData[visibleData.length - 1]?.close ?? mergedData[mergedData.length - 1]?.close ?? 1;
  const normalizedPriceMin = Number.isFinite(rawPriceMin) ? rawPriceMin : fallbackPrice * 0.99;
  const normalizedPriceMax = Number.isFinite(rawPriceMax) ? rawPriceMax : fallbackPrice * 1.01;
  const sourcePriceSpan = Math.max(normalizedPriceMax - normalizedPriceMin, Math.max(fallbackPrice * 0.001, 0.1));
  const paddedPriceMin = Math.max(0, normalizedPriceMin - sourcePriceSpan * 0.08);
  const paddedPriceMax = normalizedPriceMax + sourcePriceSpan * 0.08;
  const priceSpan = Math.max(paddedPriceMax - paddedPriceMin, 1e-9);

  const historicalVolumes = visibleData
    .filter((item) => !item.isForecast)
    .map((item) => Number(item.volume))
    .filter((value) => Number.isFinite(value) && value > 0);
  const maxVolume = Math.max(...(historicalVolumes.length ? historicalVolumes : [1]));

  const toX = (localIndex) => marginLeft + ((localIndex + 0.5) / visibleBars) * plotWidth;
  const toPriceY = (value) => marginTop + ((paddedPriceMax - value) / priceSpan) * priceHeight;
  const toVolumeY = (value) => volumeBottom - (value / maxVolume) * volumeHeight;

  const resolveHoverFromClientX = (clientX) => {
    const rect = chartSvgRef.current?.getBoundingClientRect() || chartContainerRef.current?.getBoundingClientRect();
    if (!rect || !Number.isFinite(clientX)) {
      return {
        globalIndex: visibleEnd,
        localIndex: Math.max(visibleBars - 1, 0),
        x: toX(Math.max(visibleBars - 1, 0)),
      };
    }
    const renderedWidth = Math.max(rect.width, 1);
    const rawXInViewBox = ((clientX - rect.left) / renderedWidth) * effectiveChartWidth;
    const clampedX = clamp(rawXInViewBox, marginLeft, marginLeft + plotWidth);
    const ratio = (clampedX - marginLeft) / Math.max(plotWidth, 1);
    const localIndex = clamp(
      Math.round(ratio * Math.max(visibleBars - 1, 0)),
      0,
      Math.max(visibleBars - 1, 0)
    );
    return {
      globalIndex: visibleStart + localIndex,
      localIndex,
      x: clampedX,
    };
  };

  const handlePointerMove = (clientX) => {
    if (!visibleData.length || !Number.isFinite(clientX)) {
      return;
    }
    const resolvedHover = resolveHoverFromClientX(clientX);
    setHoverState((previousHover) => {
      const hasSameIndex = previousHover.index === resolvedHover.globalIndex;
      const hasSameX =
        Number.isFinite(previousHover.x) &&
        Number.isFinite(resolvedHover.x) &&
        Math.abs(previousHover.x - resolvedHover.x) < 0.6;
      if (hasSameIndex && hasSameX) {
        return previousHover;
      }
      return {
        index: resolvedHover.globalIndex,
        x: resolvedHover.x,
      };
    });
  };

  const activeGlobalIndex =
    hoverState.index !== null && hoverState.index >= visibleStart && hoverState.index <= visibleEnd
      ? hoverState.index
      : visibleEnd;
  const activeLocalIndex = clamp(activeGlobalIndex - visibleStart, 0, Math.max(visibleBars - 1, 0));
  const activeItem = mergedData[activeGlobalIndex] ?? visibleData[visibleData.length - 1];
  const activeX =
    hoverState.index !== null &&
    hoverState.index >= visibleStart &&
    hoverState.index <= visibleEnd &&
    Number.isFinite(hoverState.x)
      ? hoverState.x
      : toX(activeLocalIndex);
  const activeCloseY = toPriceY(activeItem?.close ?? paddedPriceMin);

  const yTickCount = 6;
  const yTicks = Array.from({ length: yTickCount }, (_, index) => {
    const ratio = index / Math.max(yTickCount - 1, 1);
    const value = paddedPriceMax - ratio * priceSpan;
    return { value, y: toPriceY(value) };
  });

  const volumeTickCount = 3;
  const volumeTicks = Array.from({ length: volumeTickCount }, (_, index) => {
    const ratio = index / Math.max(volumeTickCount - 1, 1);
    const value = maxVolume - ratio * maxVolume;
    return { value, y: toVolumeY(value) };
  });

  const xTickCount = Math.min(7, visibleData.length);
  const xTickIndexes = Array.from({ length: xTickCount }, (_, index) =>
    Math.round((index * Math.max(visibleData.length - 1, 0)) / Math.max(xTickCount - 1, 1))
  ).filter((value, index, all) => all.indexOf(value) === index);

  const overlayTagWidth = Math.max(marginRight - 16, 76);

  const handleWheelZoom = (event) => {
    if (!mergedData.length || maximumVisibleBars <= minimumVisibleBars) {
      return;
    }
    if (!Number.isFinite(event.deltaY) || event.deltaY === 0) {
      return;
    }
    event.preventDefault();
    const currentVisibleBars = visibleBars;
    const zoomFactor = event.deltaY < 0 ? 0.86 : 1.14;
    const targetVisibleBars = clamp(
      Math.round(currentVisibleBars * zoomFactor),
      minimumVisibleBars,
      maximumVisibleBars
    );
    if (targetVisibleBars === currentVisibleBars) {
      return;
    }
    const resolvedHover = resolveHoverFromClientX(event.clientX);
    const anchorIndex = resolvedHover.globalIndex;
    const anchorRatio =
      currentVisibleBars <= 1
        ? 0.5
        : (anchorIndex - visibleStart) / Math.max(currentVisibleBars - 1, 1);
    let nextStart = Math.round(anchorIndex - anchorRatio * (targetVisibleBars - 1));
    nextStart = clamp(nextStart, 0, maximumVisibleBars - targetVisibleBars);
    const nextEnd = nextStart + targetVisibleBars - 1;
    setVisibleRange({
      start: nextStart,
      end: nextEnd,
    });
    setHoverState({
      index: clamp(anchorIndex, nextStart, nextEnd),
      x: null,
    });
  };

  const handleChartClick = (event) => {
    const resolvedHover = resolveHoverFromClientX(event.clientX);
    setHoverState({
      index: resolvedHover.globalIndex,
      x: resolvedHover.x,
    });
    const clickedItem = mergedData[resolvedHover.globalIndex];
    if (clickedItem?.isForecast && typeof onSelectForecastDay === "function") {
      onSelectForecastDay(clickedItem.dayAhead);
    }
  };

  return (
    <div className="chart-canvas-shell" ref={chartContainerRef}>
      <div className="chart-tooltip-strip">
        <span className="chart-tooltip-item">
          Date: <b>{formatChartTooltipDate(activeItem?.date)}</b>
        </span>
        <span className="chart-tooltip-item">
          O: <b>{formatPrice(activeItem?.open)}</b>
        </span>
        <span className="chart-tooltip-item">
          H: <b>{formatPrice(activeItem?.high)}</b>
        </span>
        <span className="chart-tooltip-item">
          L: <b>{formatPrice(activeItem?.low)}</b>
        </span>
        <span className="chart-tooltip-item">
          C: <b>{formatPrice(activeItem?.close)}</b>
        </span>
        <span className="chart-tooltip-item">
          Vol: <b>{activeItem?.isForecast ? "—" : formatCompactNumber(activeItem?.volume)}</b>
        </span>
        {activeItem?.isForecast && (
          <span className="chart-tooltip-item">
            Day: <b>D{activeItem.dayAhead}</b>
          </span>
        )}
        {activeItem?.isForecast &&
          Number.isFinite(activeItem?.uncertaintyLow) &&
          Number.isFinite(activeItem?.uncertaintyHigh) && (
            <>
              <span className="chart-tooltip-item">
                Range:{" "}
                <b>
                  {formatPrice(activeItem?.uncertaintyLow)} → {formatPrice(activeItem?.uncertaintyHigh)}
                </b>
              </span>
              <span className="chart-tooltip-item">
                Spread:{" "}
                <b>
                  ±{formatPrice(activeItem?.uncertaintySpread)}{" "}
                  {Number.isFinite(activeItem?.uncertaintySpreadPct)
                    ? `(${activeItem.uncertaintySpreadPct.toFixed(2)}%)`
                    : ""}
                </b>
              </span>
            </>
          )}
      </div>
      <svg
        className="chart-svg"
        ref={chartSvgRef}
        width={effectiveChartWidth}
        height={chartHeight}
        viewBox={`0 0 ${effectiveChartWidth} ${chartHeight}`}
      >
        <rect
          x={marginLeft}
          y={marginTop}
          width={plotWidth}
          height={priceHeight}
          rx={10}
          className="chart-surface-price"
        />
        <rect
          x={marginLeft}
          y={volumeTop}
          width={plotWidth}
          height={volumeHeight}
          rx={10}
          className="chart-surface-volume"
        />

        {yTicks.map((tick, index) => (
          <g key={`price-grid-${index}`}>
            <line
              x1={marginLeft}
              y1={tick.y}
              x2={marginLeft + plotWidth}
              y2={tick.y}
              className={index === yTicks.length - 1 ? "chart-grid-line chart-grid-line-strong" : "chart-grid-line"}
            />
            <text x={marginLeft - 8} y={tick.y + 3} textAnchor="end" className="chart-axis-label">
              {formatAxisPrice(tick.value)}
            </text>
          </g>
        ))}

        {volumeTicks.map((tick, index) => (
          <g key={`volume-grid-${index}`}>
            <line
              x1={marginLeft}
              y1={tick.y}
              x2={marginLeft + plotWidth}
              y2={tick.y}
              className="chart-grid-line"
            />
            <text x={marginLeft - 8} y={tick.y + 3} textAnchor="end" className="chart-axis-label">
              {formatCompactNumber(tick.value)}
            </text>
          </g>
        ))}

        {xTickIndexes.map((tickIndex) => {
          const x = toX(tickIndex);
          const tickDate = visibleData[tickIndex]?.date;
          return (
            <g key={`x-grid-${tickIndex}`}>
              <line x1={x} y1={marginTop} x2={x} y2={volumeBottom} className="chart-grid-line chart-grid-line-vertical" />
              <line x1={x} y1={volumeBottom} x2={x} y2={volumeBottom + 4} className="chart-axis-line" />
              <text x={x} y={chartHeight - 8} textAnchor="middle" className="chart-axis-label">
                {formatDate(tickDate)}
              </text>
            </g>
          );
        })}

        {overlayItems.map((overlay) => {
          const y = toPriceY(overlay.value);
          return (
            <g key={`overlay-${overlay.key}`}>
              <line
                x1={marginLeft}
                y1={y}
                x2={marginLeft + plotWidth}
                y2={y}
                stroke={overlay.color}
                strokeWidth={1.2}
                strokeDasharray={overlay.dash}
              />
              <rect
                x={marginLeft + plotWidth + 8}
                y={y - 9}
                width={overlayTagWidth}
                height={18}
                rx={5}
                className="chart-overlay-tag-bg"
              />
              <text
                x={marginLeft + plotWidth + 14}
                y={y + 3.5}
                className="chart-overlay-tag-text"
              >
                {overlay.label} {formatAxisPrice(overlay.value)}
              </text>
            </g>
          );
        })}

        {visibleData.map((item, localIndex) => {
          if (!item.isForecast) {
            return null;
          }
          const lower = Number(item?.uncertaintyLow);
          const upper = Number(item?.uncertaintyHigh);
          if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
            return null;
          }
          const x = toX(localIndex);
          const topY = toPriceY(Math.max(lower, upper));
          const bottomY = toPriceY(Math.min(lower, upper));
          const height = Math.max(bottomY - topY, 1.2);
          const width = clamp(candleSlotWidth * 1.08, 5, 26);
          return (
            <rect
              key={`spread-${item.id}`}
              x={x - width / 2}
              y={topY}
              width={width}
              height={height}
              rx={2}
              className={`chart-forecast-spread ${item.isSelected ? "chart-forecast-spread-selected" : ""}`}
            />
          );
        })}

        {visibleData.map((item, localIndex) => {
          const x = toX(localIndex);
          const openY = toPriceY(item.open);
          const closeY = toPriceY(item.close);
          const highY = toPriceY(item.high);
          const lowY = toPriceY(item.low);
          const bodyTop = Math.min(openY, closeY);
          const bodyHeight = Math.max(Math.abs(closeY - openY), 1.4);
          const isBullish = item.close >= item.open;
          const bodyFill = item.isForecast
            ? item.isSelected
              ? "rgba(69, 133, 136, 0.58)"
              : "rgba(69, 133, 136, 0.33)"
            : isBullish
              ? "rgba(152, 151, 26, 0.92)"
              : "rgba(204, 36, 29, 0.92)";
          const strokeColor = item.isForecast
            ? item.isSelected
              ? "#076678"
              : "#458588"
            : isBullish
              ? "#79740e"
              : "#9d0006";

          return (
            <g key={item.id}>
              <line
                x1={x}
                y1={highY}
                x2={x}
                y2={lowY}
                stroke={strokeColor}
                strokeWidth={1.2}
                strokeDasharray={item.isForecast ? "4 3" : undefined}
              />
              <rect
                x={x - candleBodyWidth / 2}
                y={bodyTop}
                width={candleBodyWidth}
                height={bodyHeight}
                rx={1.8}
                fill={bodyFill}
                stroke={strokeColor}
                strokeWidth={item.isSelected ? 1.8 : 1}
                strokeDasharray={item.isForecast ? "4 3" : undefined}
              />
            </g>
          );
        })}

        {visibleData.map((item, localIndex) => {
          if (item.isForecast || !Number.isFinite(item.volume) || item.volume <= 0) {
            return null;
          }
          const x = toX(localIndex);
          const y = toVolumeY(item.volume);
          const height = Math.max(volumeBottom - y, 1.2);
          const barWidth = clamp(candleSlotWidth * 0.66, 2.5, 14);
          const fill = item.close >= item.open ? "rgba(152, 151, 26, 0.52)" : "rgba(204, 36, 29, 0.52)";
          return (
            <rect
              key={`volume-${item.id}`}
              x={x - barWidth / 2}
              y={y}
              width={barWidth}
              height={height}
              rx={1}
              fill={fill}
            />
          );
        })}

        <line x1={marginLeft} y1={priceBottom} x2={marginLeft + plotWidth} y2={priceBottom} className="chart-axis-line" />
        <line x1={marginLeft} y1={volumeBottom} x2={marginLeft + plotWidth} y2={volumeBottom} className="chart-axis-line" />
        <line x1={marginLeft} y1={marginTop} x2={marginLeft} y2={volumeBottom} className="chart-axis-line" />
        <line
          x1={marginLeft + plotWidth}
          y1={marginTop}
          x2={marginLeft + plotWidth}
          y2={priceBottom}
          className="chart-axis-line"
        />

        <line
          x1={activeX}
          y1={marginTop}
          x2={activeX}
          y2={volumeBottom}
          className="chart-hover-line"
        />
        <line
          x1={marginLeft}
          y1={activeCloseY}
          x2={marginLeft + plotWidth}
          y2={activeCloseY}
          className="chart-hover-line chart-hover-line-horizontal"
        />
        <circle cx={activeX} cy={activeCloseY} r={3.2} className="chart-hover-marker" />

        <rect
          x={marginLeft}
          y={marginTop}
          width={plotWidth}
          height={priceHeight + panelGap + volumeHeight}
          className="chart-interaction-layer"
          onMouseMove={(event) => handlePointerMove(event.clientX)}
          onMouseLeave={() => setHoverState({ index: null, x: null })}
          onTouchStart={(event) => handlePointerMove(event.touches?.[0]?.clientX)}
          onTouchMove={(event) => handlePointerMove(event.touches?.[0]?.clientX)}
          onTouchEnd={() => setHoverState({ index: null, x: null })}
          onWheel={handleWheelZoom}
          onClick={handleChartClick}
        />
      </svg>
      {showTechnical && (
        <p className="chart-interaction-note">
          Hover updates OHLC/volume in real time. Wheel zoom changes scale. Forecast spread band shows uncertainty range.
        </p>
      )}
    </div>
  );
}

function TradingTerminalWidget({ role = "broker", onRoleChange }) {
  const isAdmin = role === "admin";
  const [symbol, setSymbol] = useState("BTC/USD");
  const [forecastDays, setForecastDays] = useState(14);
  const [modelType, setModelType] = useState("xgboost");
  const [market, setMarket] = useState(null);
  const [liveMarket, setLiveMarket] = useState(null);
  const [liveEnabled, setLiveEnabled] = useState(true);
  const [liveIntervalSec, setLiveIntervalSec] = useState(10);
  const [lastLiveUpdateAt, setLastLiveUpdateAt] = useState(null);
  const [isLiveUpdating, setIsLiveUpdating] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [selectedForecastDay, setSelectedForecastDay] = useState(null);
  const [loadingMarket, setLoadingMarket] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [featurePreview, setFeaturePreview] = useState(null);
  const [featureSearch, setFeatureSearch] = useState("");
  const [featureCategory, setFeatureCategory] = useState("all");
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [error, setError] = useState("");
  const liveFetchInFlightRef = useRef(false);

  const candles = market?.candles ?? [];
  const liveCandles = liveMarket?.candles ?? [];
  const spotCandles = liveCandles.length ? liveCandles : candles;
  const freshnessSource = liveMarket ?? market;
  const currentPrice = spotCandles.length ? spotCandles[spotCandles.length - 1].close : null;
  const previousPrice = spotCandles.length > 1 ? spotCandles[spotCandles.length - 2].close : null;
  const priceDelta = currentPrice !== null && previousPrice !== null ? currentPrice - previousPrice : null;
  const priceDeltaPct =
    priceDelta !== null && previousPrice
      ? (priceDelta / previousPrice) * 100
      : null;

  const confidence = useMemo(() => {
    if (!prediction?.metrics?.mape && prediction?.metrics?.mape !== 0) {
      return 0;
    }
    const raw = Math.max(0, 1 - prediction.metrics.mape);
    return Math.round(raw * 100);
  }, [prediction]);
  async function refreshMarket(options = {}) {
    const { silent = false } = options;
    try {
      if (!silent) {
        setLoadingMarket(true);
        setError("");
      }
      const payload = await getCandles({ symbol, timeframe: "1d", limit: 70 });
      setMarket(payload);
    } catch (err) {
      if (!silent) {
        setError(err.message || "Failed to load market data");
      }
    } finally {
      if (!silent) {
        setLoadingMarket(false);
      }
    }
  }
  async function refreshFeaturePreview(options = {}) {
    const { silent = false } = options;
    try {
      if (!silent) {
        setLoadingFeatures(true);
      }
      const requestedForecastDays = Math.max(
        MIN_FORECAST_DAYS,
        Math.min(MAX_FORECAST_DAYS, Number(forecastDays) || MIN_FORECAST_DAYS)
      );
      const payload = await getFeaturePreview({
        symbol,
        timeframe: "1d",
        forecastDays: requestedForecastDays,
        source: "exchange",
        exchangeLimit: 700,
        historySize: 72,
      });
      setFeaturePreview(payload);
    } catch (err) {
      if (!silent) {
        setError(err.message || "Failed to load feature preview");
      }
    } finally {
      if (!silent) {
        setLoadingFeatures(false);
      }
    }
  }

  function handleForecastDaysChange(rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) {
      setForecastDays(MIN_FORECAST_DAYS);
      return;
    }
    const normalized = Math.round(parsed);
    setForecastDays(Math.max(MIN_FORECAST_DAYS, Math.min(MAX_FORECAST_DAYS, normalized)));
  }


  async function refreshLiveMarket(options = {}) {
    const { silent = true } = options;
    if (liveFetchInFlightRef.current) {
      return;
    }
    liveFetchInFlightRef.current = true;
    try {
      if (!silent) {
        setIsLiveUpdating(true);
      }
      const payload = await getCandles({ symbol, timeframe: "1m", limit: 120 });
      setLiveMarket(payload);
      setLastLiveUpdateAt(new Date().toISOString());
    } catch (err) {
      if (!silent) {
        setError(err.message || "Failed to load live market data");
      }
    } finally {
      liveFetchInFlightRef.current = false;
      if (!silent) {
        setIsLiveUpdating(false);
      }
    }
  }

  function refreshAllMarketData() {
    refreshMarket();
    refreshLiveMarket({ silent: false });
    refreshFeaturePreview({ silent: true });
  }
  async function runPredict() {
    try {
      setLoadingPrediction(true);
      setError("");
      const requestedForecastDays = Math.max(
        MIN_FORECAST_DAYS,
        Math.min(MAX_FORECAST_DAYS, Number(forecastDays) || MIN_FORECAST_DAYS)
      );
      setForecastDays(requestedForecastDays);
      const payload = await getPrediction({
        symbol,
        forecastDays: requestedForecastDays,
        timeframe: "1d",
        modelType,
      });
      setPrediction(payload);
      setSelectedForecastDay(payload?.daily_path?.[0]?.day_ahead ?? null);
      refreshFeaturePreview({ silent: true });
    } catch (err) {
      setError(err.message || "Prediction request failed");
    } finally {
      setLoadingPrediction(false);
    }
  }

  useEffect(() => {
    setSelectedForecastDay(null);
    setPrediction(null);
    setLiveMarket(null);
    refreshMarket();
    refreshLiveMarket({ silent: false });
    refreshFeaturePreview({ silent: false });
  }, [symbol]);

  useEffect(() => {
    refreshFeaturePreview({ silent: true });
  }, [forecastDays]);

  useEffect(() => {
    if (!selectedFeature) {
      return;
    }
    const freshVersion = (featurePreview?.features ?? []).find(
      (item) => item.name === selectedFeature.name
    );
    if (!freshVersion) {
      setSelectedFeature(null);
      return;
    }
    if (freshVersion !== selectedFeature) {
      setSelectedFeature(freshVersion);
    }
  }, [featurePreview, selectedFeature]);

  useEffect(() => {
    if (!selectedFeature) {
      return undefined;
    }
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setSelectedFeature(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedFeature]);

  useEffect(() => {
    if (!isAdmin) {
      setSelectedFeature(null);
    }
  }, [isAdmin]);

  useEffect(() => {
    if (!liveEnabled) {
      return undefined;
    }
    const intervalId = setInterval(() => {
      refreshLiveMarket({ silent: true });
    }, liveIntervalSec * 1000);
    return () => clearInterval(intervalId);
  }, [symbol, liveEnabled, liveIntervalSec]);

  const liveRows = (liveCandles.length ? liveCandles : candles).slice(-5).reverse();
  const forecastPath = prediction?.daily_path ?? [];
  const selectedForecastPoint =
    forecastPath.find((point) => point.day_ahead === selectedForecastDay) ??
    forecastPath[0] ??
    null;
  const selectedForecastDelta =
    selectedForecastPoint && currentPrice !== null
      ? selectedForecastPoint.predicted_price - currentPrice
      : null;
  const selectedForecastDeltaPct =
    selectedForecastDelta !== null && currentPrice
      ? (selectedForecastDelta / currentPrice) * 100
      : null;
  const recentRangePct = useMemo(() => {
    const sourceCandles = spotCandles.length ? spotCandles : candles;
    const recentSlice = sourceCandles.slice(-20);
    const ranges = recentSlice
      .map((item) => {
        const high = Number(item?.high);
        const low = Number(item?.low);
        const close = Number(item?.close);
        if (!Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close) || close <= 0) {
          return null;
        }
        return ((high - low) / close) * 100;
      })
      .filter((value) => value !== null && Number.isFinite(value));
    if (!ranges.length) {
      return null;
    }
    return ranges.reduce((sum, value) => sum + value, 0) / ranges.length;
  }, [spotCandles, candles]);
  const forecastSpreadPath = useMemo(() => {
    if (!forecastPath.length) {
      return [];
    }
    const mapeRatioRaw = Number(prediction?.metrics?.mape);
    const rmseValue = Number(prediction?.metrics?.rmse);
    const volatilityRatioRaw = Number(recentRangePct) / 100;
    const spreadTightness = 0.52;
    const maxSpreadRatio = 0.038;
    const minSpreadRatio = 0.00045;
    return forecastPath
      .map((point, index) => {
        const predictedPrice = Number(point?.predicted_price);
        if (!Number.isFinite(predictedPrice) || predictedPrice <= 0) {
          return null;
        }
        const dayAhead = Number(point?.day_ahead ?? index + 1);
        const horizonScale = clamp(
          Math.sqrt(dayAhead / Math.max(Number(forecastDays) || 1, 1)),
          0.75,
          1.75
        );
        const mapeSpreadRatio = Number.isFinite(mapeRatioRaw)
          ? clamp(mapeRatioRaw * 0.38, 0.0012, 0.085) * horizonScale
          : 0;
        const volatilitySpreadRatio = Number.isFinite(volatilityRatioRaw)
          ? clamp(volatilityRatioRaw * 0.22, 0.001, 0.06) * horizonScale
          : 0;
        const baseRatioSpread = predictedPrice * Math.max(mapeSpreadRatio, volatilitySpreadRatio, 0.0018);
        const rmseSpread = Number.isFinite(rmseValue) ? rmseValue * clamp(horizonScale * 0.6, 0.45, 1.15) : 0;
        const rawSpread = Math.max(baseRatioSpread, rmseSpread, predictedPrice * 0.0009);
        const tightenedSpread = rawSpread * spreadTightness;
        const spread = clamp(tightenedSpread, predictedPrice * minSpreadRatio, predictedPrice * maxSpreadRatio);
        const lower = Math.max(0, predictedPrice - spread);
        const upper = predictedPrice + spread;
        return {
          dayAhead,
          lower,
          upper,
          spread,
          spreadPct: (spread / predictedPrice) * 100,
        };
      })
      .filter(Boolean);
  }, [forecastPath, prediction?.metrics?.mape, prediction?.metrics?.rmse, recentRangePct, forecastDays]);
  const volatilityProfile = useMemo(() => {
    if (recentRangePct === null) {
      return {
        label: "No volatility profile",
        tone: "neutral",
      };
    }
    if (recentRangePct >= HIGH_VOLATILITY_THRESHOLD_PCT) {
      return {
        label: "High volatility",
        tone: "down",
      };
    }
    if (recentRangePct <= LOW_VOLATILITY_THRESHOLD_PCT) {
      return {
        label: "Calm market",
        tone: "up",
      };
    }
    return {
      label: "Balanced volatility",
      tone: "neutral",
    };
  }, [recentRangePct]);
  const forecastPathStats = useMemo(() => {
    if (!forecastPath.length || currentPrice === null || currentPrice === undefined) {
      return {
        upDays: 0,
        downDays: 0,
        flatDays: 0,
        pathHighPct: null,
        pathLowPct: null,
        biasLabel: "No path bias",
      };
    }
    let upDays = 0;
    let downDays = 0;
    let flatDays = 0;
    let previousPrice = Number(currentPrice);
    const prices = [];
    forecastPath.forEach((point) => {
      const price = Number(point?.predicted_price);
      if (!Number.isFinite(price)) {
        return;
      }
      prices.push(price);
      if (price > previousPrice) {
        upDays += 1;
      } else if (price < previousPrice) {
        downDays += 1;
      } else {
        flatDays += 1;
      }
      previousPrice = price;
    });
    if (!prices.length) {
      return {
        upDays: 0,
        downDays: 0,
        flatDays: 0,
        pathHighPct: null,
        pathLowPct: null,
        biasLabel: "No path bias",
      };
    }
    const pathHigh = Math.max(...prices);
    const pathLow = Math.min(...prices);
    const pathHighPct = Number(currentPrice) > 0 ? ((pathHigh - Number(currentPrice)) / Number(currentPrice)) * 100 : null;
    const pathLowPct = Number(currentPrice) > 0 ? ((pathLow - Number(currentPrice)) / Number(currentPrice)) * 100 : null;
    const upShare = (upDays / prices.length) * 100;
    const downShare = (downDays / prices.length) * 100;
    const biasLabel =
      upShare >= 60
        ? "Path bias: bullish"
        : downShare >= 60
          ? "Path bias: bearish"
          : "Path bias: mixed";
    return {
      upDays,
      downDays,
      flatDays,
      pathHighPct,
      pathLowPct,
      biasLabel,
    };
  }, [forecastPath, currentPrice]);
  const decisionSummary = useMemo(() => {
    if (!selectedForecastPoint || selectedForecastDeltaPct === null || selectedForecastDelta === null) {
      return {
        action: "WAIT",
        tone: "neutral",
        confidenceBand: "No signal",
        rationale: "Run prediction to produce a trade direction.",
        expectedMoveLabel: "—",
      };
    }
    const absMovePct = Math.abs(selectedForecastDeltaPct);
    let action = "HOLD";
    let tone = "neutral";
    if (absMovePct >= HOLD_MOVE_THRESHOLD_PCT) {
      if (selectedForecastDeltaPct > 0) {
        action = "BUY";
        tone = "up";
      } else {
        action = "SELL";
        tone = "down";
      }
    }
    let confidenceBand = "Low conviction";
    if (confidence >= 75) {
      confidenceBand = "High conviction";
    } else if (confidence >= 55) {
      confidenceBand = "Medium conviction";
    }
    if (absMovePct < HOLD_MOVE_THRESHOLD_PCT) {
      confidenceBand = `${confidenceBand} · low move`;
    } else if (absMovePct >= STRONG_MOVE_THRESHOLD_PCT) {
      confidenceBand = `${confidenceBand} · strong move`;
    }
    const directionWord = selectedForecastDeltaPct > 0 ? "upside" : "downside";
    const rationale =
      action === "HOLD"
        ? "Expected move vs spot is small; wait for a stronger setup."
        : `Projected ${directionWord} by day ${selectedForecastPoint.day_ahead}; align position with direction.`;
    return {
      action,
      tone,
      confidenceBand,
      rationale,
      expectedMoveLabel: `${selectedForecastDelta >= 0 ? "+" : ""}${formatPrice(selectedForecastDelta)} (${selectedForecastDeltaPct.toFixed(2)}%)`,
    };
  }, [selectedForecastDelta, selectedForecastDeltaPct, selectedForecastPoint, confidence]);
  const tradePlan = useMemo(() => {
    const entry = Number(currentPrice);
    const target = Number(selectedForecastPoint?.predicted_price);
    if (!Number.isFinite(entry) || entry <= 0) {
      return null;
    }
    if (!Number.isFinite(target)) {
      return {
        entry,
        target: null,
        stop: null,
        riskReward: null,
        stopDistancePct: null,
      };
    }
    const computedStopDistancePct = Math.max(
      MIN_STOP_DISTANCE_PCT,
      Math.min(MAX_STOP_DISTANCE_PCT, (recentRangePct ?? MIN_STOP_DISTANCE_PCT) * 0.8)
    );
    const stopDistance = (entry * computedStopDistancePct) / 100;
    const stop =
      decisionSummary.action === "BUY"
        ? entry - stopDistance
        : decisionSummary.action === "SELL"
          ? entry + stopDistance
          : null;
    const riskDistance = stop === null ? null : Math.abs(stop - entry);
    const rewardDistance = Math.abs(target - entry);
    const riskReward =
      riskDistance === null || riskDistance === 0 ? null : rewardDistance / riskDistance;
    return {
      entry,
      target,
      stop,
      riskReward,
      stopDistancePct: computedStopDistancePct,
    };
  }, [currentPrice, selectedForecastPoint, recentRangePct, decisionSummary.action]);
  const brokerLevelOverlays = useMemo(() => {
    if (isAdmin || !tradePlan) {
      return [];
    }
    const overlays = [];
    if (Number.isFinite(tradePlan.entry)) {
      overlays.push({
        key: "entry-level",
        label: "Entry",
        value: tradePlan.entry,
        tone: "neutral",
      });
    }
    if (Number.isFinite(tradePlan.target)) {
      overlays.push({
        key: "target-level",
        label: "Target",
        value: tradePlan.target,
        tone: tradePlan.target >= tradePlan.entry ? "up" : "down",
      });
    }
    if (Number.isFinite(tradePlan.stop)) {
      overlays.push({
        key: "stop-level",
        label: "Stop",
        value: tradePlan.stop,
        tone: "down",
      });
    }
    return overlays;
  }, [tradePlan, isAdmin]);
  const trainingWindows = prediction?.training_data_windows ?? [];
  const requestWindows = prediction?.request_data_windows ?? [];
  const lossLabel = prediction?.loss_function ?? "—";
  const scoringLabel = prediction?.tuning_scoring ?? "—";
  const modelTypeLabel = prediction?.model_type ?? "—";
  const modelIdLabel = prediction?.model_id ?? "—";
  const trainedAtLabel = formatDateTime(prediction?.model_trained_at);
  const mapeValue = prediction?.metrics?.mape;
  const maeValue = prediction?.metrics?.mae;
  const rmseValue = prediction?.metrics?.rmse;
  const mseValue = prediction?.metrics?.mse;
  const maeRelativePct =
    maeValue !== undefined && maeValue !== null && currentPrice
      ? (maeValue / currentPrice) * 100
      : null;
  const rmseRelativePct =
    rmseValue !== undefined && rmseValue !== null && currentPrice
      ? (rmseValue / currentPrice) * 100
      : null;
  const featureItems = featurePreview?.features ?? [];
  const featureCategories = useMemo(() => {
    const map = new Map();
    featureItems.forEach((item) => {
      const key = item?.category || "other";
      map.set(key, (map.get(key) || 0) + 1);
    });
    return Array.from(map.entries())
      .map(([value, count]) => ({ value, count }))
      .sort((a, b) => a.value.localeCompare(b.value));
  }, [featureItems]);
  const visibleFeatureItems = useMemo(() => {
    const search = featureSearch.trim().toLowerCase();
    return featureItems.filter((item) => {
      if (!item) {
        return false;
      }
      if (featureCategory !== "all" && item.category !== featureCategory) {
        return false;
      }
      if (!search) {
        return true;
      }
      return item.name.toLowerCase().includes(search);
    });
  }, [featureItems, featureCategory, featureSearch]);
  const selectedFeatureRangeValue =
    selectedFeature?.max_value !== null &&
    selectedFeature?.max_value !== undefined &&
    selectedFeature?.min_value !== null &&
    selectedFeature?.min_value !== undefined
      ? selectedFeature.max_value - selectedFeature.min_value
      : null;
  const fieldHints = useMemo(() => {
    const horizonInterpretation =
      forecastDays >= 21
        ? "Длинный горизонт: ожидайте выше ошибку и более широкий разброс сценариев."
        : forecastDays >= 8
          ? "Средний горизонт: компромисс между стабильностью и дальностью прогноза."
          : "Короткий горизонт: обычно более предсказуем, чем long-horizon.";
    const trainRowsValue = prediction?.train_rows;
    const trainRowsInterpretation =
      trainRowsValue === undefined || trainRowsValue === null
        ? "Запустите прогноз, чтобы увидеть объем обучающих данных."
        : trainRowsValue < 350
          ? "Для крипто это относительно небольшой train-set; модель может быть чувствительна к шуму."
          : "Объем данных приемлемый для базового daily-прогноза.";
    const maeInterpretation =
      maeRelativePct === null
        ? "Сопоставляйте MAE с текущей ценой, чтобы понимать масштаб ошибки."
        : `Сейчас MAE ≈ ${formatPercent(maeRelativePct)} от текущей цены. Чем ниже этот процент, тем надежнее прогноз.`;
    const rmseInterpretation =
      rmseRelativePct === null
        ? "Сравнивайте RMSE с MAE: если RMSE заметно выше, у модели есть редкие, но крупные промахи."
        : `Сейчас RMSE ≈ ${formatPercent(rmseRelativePct)} от текущей цены. Существенно выше MAE = повышенная чувствительность к выбросам.`;
    const mapeInterpretation =
      mapeValue === undefined || mapeValue === null
        ? "MAPE в процентах появится после прогноза."
        : `Сейчас MAPE = ${formatPercent(mapeValue * 100)}. Чем ниже процент, тем лучше относительная точность.`;
    const mseInterpretation =
      mseValue === undefined || mseValue === null
        ? "MSE появится после прогноза."
        : `Сейчас MSE = ${formatMetricValue(mseValue)}. Рост MSE означает увеличение крупных ошибок.`;
    const trainingLag = prediction?.training_max_data_lag_minutes;
    const requestLag = prediction?.request_max_data_lag_minutes;
    return {
      ...BASE_FIELD_HINTS,
      lastClose: `${BASE_FIELD_HINTS.lastClose}\nСейчас: ${formatPrice(currentPrice)}.\nИнтерпретация: сравнивайте все прогнозные Δ именно с этой ценой.`,
      forecastHorizon: `${BASE_FIELD_HINTS.forecastHorizon}\nСейчас: ${forecastDays} дней.\n${horizonInterpretation}`,
      trainRows: `${BASE_FIELD_HINTS.trainRows}\nСейчас: ${trainRowsValue ?? "—"}.\n${trainRowsInterpretation}`,
      mae: `${BASE_FIELD_HINTS.mae}\nСейчас: ${maeValue?.toFixed(2) ?? "—"} USD.\n${maeInterpretation}`,
      rmse: `${BASE_FIELD_HINTS.rmse}\nСейчас: ${rmseValue?.toFixed(2) ?? "—"} USD.\n${rmseInterpretation}`,
      mape: `${BASE_FIELD_HINTS.mape}\nСейчас: ${mapeValue === undefined || mapeValue === null ? "—" : formatPercent(mapeValue * 100)}.\n${mapeInterpretation}`,
      mse: `${BASE_FIELD_HINTS.mse}\nСейчас: ${mseValue === undefined || mseValue === null ? "—" : formatMetricValue(mseValue)}.\n${mseInterpretation}`,
      modelId: `${BASE_FIELD_HINTS.modelId}\nСейчас: ${modelIdLabel}. Для текущей проблемы важно, что это single-model, а не universal.`,
      modelType: `${BASE_FIELD_HINTS.modelType}\nСейчас: ${modelTypeLabel}.`,
      lossFunction: `${BASE_FIELD_HINTS.lossFunction}\nСейчас: ${lossLabel}. Это корректно для вашего кейса с выбросами.`,
      tuningScoring: `${BASE_FIELD_HINTS.tuningScoring}\nСейчас: ${scoringLabel}. Для вашей задачи это означает, что подбор оптимизирует абсолютную ошибку в USD.`,
      trainedAt: `${BASE_FIELD_HINTS.trainedAt}\nСейчас: ${trainedAtLabel}.`,
      trainingDataFreshness: `${BASE_FIELD_HINTS.trainingDataFreshness}\nСейчас: ${prediction?.training_data_is_fresh ? "Fresh" : "Delayed"}.`,
      trainingMaxLag: `${BASE_FIELD_HINTS.trainingMaxLag}\nСейчас: ${formatLagMinutes(trainingLag)}.`,
      requestDataFreshness: `${BASE_FIELD_HINTS.requestDataFreshness}\nСейчас: ${prediction?.request_data_is_fresh ? "Fresh" : "Delayed"}.`,
      requestMaxLag: `${BASE_FIELD_HINTS.requestMaxLag}\nСейчас: ${formatLagMinutes(requestLag)}.`,
    };
  }, [
    currentPrice,
    forecastDays,
    prediction?.train_rows,
    prediction?.training_data_is_fresh,
    prediction?.training_max_data_lag_minutes,
    prediction?.request_data_is_fresh,
    prediction?.request_max_data_lag_minutes,
    maeValue,
    maeRelativePct,
    rmseValue,
    rmseRelativePct,
    mapeValue,
    mseValue,
    modelIdLabel,
    modelTypeLabel,
    lossLabel,
    scoringLabel,
    trainedAtLabel,
  ]);
  const confidenceHint = useMemo(() => {
    if (mapeValue === undefined || mapeValue === null) {
      return (
        "Confidence рассчитывается из MAPE по формуле: max(0, 1 - MAPE) * 100.\n" +
        "Запустите прогноз, чтобы увидеть интерпретацию для текущего состояния модели."
      );
    }
    const base = `Формула confidence: max(0, 1 - MAPE) * 100.\nСейчас MAPE: ${mapeValue.toFixed(4)} (${formatPercent(
      mapeValue * 100
    )}), confidence: ${confidence}%.`;
    if (confidence === 0 && mapeValue >= 1) {
      return (
        `${base}\n` +
        "Интерпретация текущей проблемы: 0% получается из формулы (MAPE > 1), а не обязательно из-за поломки модели.\n" +
        "Для практической оценки ориентируйтесь дополнительно на MAE и на прогнозные Δ по дням."
      );
    }
    return `${base}\nИнтерпретация: чем ниже MAPE, тем выше confidence по текущей формуле.`;
  }, [mapeValue, confidence]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Decision Intelligence Workspace</p>
          <h1 className="title">Broker Decision Platform</h1>
          <p className="topbar-subtitle">
            {isAdmin
              ? "Admin view exposes diagnostic controls and full model context."
              : "Broker view keeps only the inputs and outputs needed for a trading decision."}
          </p>
        </div>
        <div className="controls">
          <div className="role-switch" role="group" aria-label="Role selector">
            <span className="role-switch-label">Role</span>
            <div className="role-switch-options">
              {ROLE_OPTIONS.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  className={`role-switch-btn ${role === item.value ? "is-active" : ""}`}
                  onClick={() => onRoleChange?.(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <select
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            className="control-select"
          >
            {SYMBOLS.map((item) => (
              <option value={item} key={item}>
                {item}
              </option>
            ))}
          </select>
          <select
            value={modelType}
            onChange={(event) => setModelType(event.target.value)}
            className="control-select"
          >
            {MODEL_OPTIONS.map((item) => (
              <option value={item.value} key={item.value}>
                {item.label}
              </option>
            ))}
          </select>
          <label className="horizon-input-wrap">
            <span>Horizon (days)</span>
            <input
              type="number"
              min={MIN_FORECAST_DAYS}
              max={MAX_FORECAST_DAYS}
              step={1}
              value={forecastDays}
              onChange={(event) => handleForecastDaysChange(event.target.value)}
              className="control-input"
            />
          </label>
          {isAdmin && (
            <label className="live-toggle-wrap">
              <input
                type="checkbox"
                checked={liveEnabled}
                onChange={(event) => setLiveEnabled(event.target.checked)}
              />
              <span>Live</span>
            </label>
          )}
          {isAdmin && (
            <label className="horizon-input-wrap">
              <span>Live interval</span>
              <select
                value={liveIntervalSec}
                onChange={(event) => setLiveIntervalSec(Number(event.target.value) || 10)}
                className="control-select"
                disabled={!liveEnabled}
              >
                {LIVE_INTERVAL_OPTIONS.map((seconds) => (
                  <option key={`live-${seconds}`} value={seconds}>
                    {seconds}s
                  </option>
                ))}
              </select>
            </label>
          )}
          {isAdmin && (
            <button
              className="btn btn-muted"
              onClick={refreshAllMarketData}
              disabled={loadingMarket || isLiveUpdating}
            >
              {loadingMarket ? "Refreshing..." : "Refresh"}
            </button>
          )}
          <button className="btn btn-primary" onClick={runPredict} disabled={loadingPrediction}>
            {loadingPrediction ? "Updating signal..." : "Update signal"}
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className={`layout-grid ${isAdmin ? "" : "layout-grid-broker"}`}>
        <section className="main-column">
          <div className="panel hero-panel">
            <div>
              <p className="eyebrow">Spot Price</p>
              <h2 className="hero-price">{formatPrice(currentPrice)}</h2>
              <p
                className={`hero-delta ${
                  priceDelta !== null && priceDelta >= 0 ? "text-up" : "text-down"
                }`}
              >
                {priceDelta !== null ? `${formatPrice(priceDelta)} (${priceDeltaPct?.toFixed(2)}%)` : "—"}
              </p>
            </div>
            <div className="freshness-box">
              <p className="fresh-label">{isAdmin ? "Data freshness · 1m live" : "Data freshness"}</p>
              <p className={`fresh-state ${freshnessSource?.is_fresh ? "text-up" : "text-down"}`}>
                {freshnessSource?.is_fresh ? "Fresh" : "Delayed"}
              </p>
              {isAdmin && (
                <>
                  <p className="fresh-meta">
                    lag {freshnessSource?.data_lag_minutes?.toFixed(1) ?? "—"} min
                  </p>
                  <p className="fresh-meta">
                    last candle:{" "}
                    {freshnessSource?.last_candle_at
                      ? new Date(freshnessSource.last_candle_at).toLocaleString()
                      : "—"}
                  </p>
                  <p className="fresh-meta">
                    live mode:{" "}
                    {liveEnabled
                      ? `ON · ${liveIntervalSec}s${isLiveUpdating ? " · updating..." : ""}`
                      : "OFF"}
                  </p>
                </>
              )}
              <p className="fresh-meta">
                last live sync:{" "}
                {lastLiveUpdateAt ? new Date(lastLiveUpdateAt).toLocaleTimeString() : "—"}
              </p>
            </div>
          </div>

          <div className={`panel decision-panel decision-panel-${decisionSummary.tone}`}>
            <div className="decision-panel-head">
              <p className="eyebrow">Decision Signal</p>
              <span className={`decision-badge decision-badge-${decisionSummary.tone}`}>
                {decisionSummary.action}
              </span>
            </div>
            <div className="decision-grid">
              <div className="decision-main">
                <p className="decision-value">{decisionSummary.expectedMoveLabel}</p>
                <p className="decision-rationale">{decisionSummary.rationale}</p>
              </div>
              <div className="decision-meta">
                <p>
                  Conviction: <b>{decisionSummary.confidenceBand}</b>
                </p>
                <p>
                  Volatility:{" "}
                  <b className={volatilityProfile.tone === "up" ? "text-up" : volatilityProfile.tone === "down" ? "text-down" : ""}>
                    {volatilityProfile.label}
                  </b>
                </p>
                <p>
                  Focus day:{" "}
                  <b>
                    {selectedForecastPoint
                      ? `D${selectedForecastPoint.day_ahead} · ${formatDate(selectedForecastPoint.predict_for_at)}`
                      : "—"}
                  </b>
                </p>
                <p>
                  {forecastPathStats.biasLabel}:{" "}
                  <b>
                    {forecastPathStats.upDays}↑ / {forecastPathStats.downDays}↓ / {forecastPathStats.flatDays}→
                  </b>
                </p>
                <p>
                  Target price:{" "}
                  <b>{selectedForecastPoint ? formatPrice(selectedForecastPoint.predicted_price) : "—"}</b>
                </p>
              </div>
            </div>
          </div>
          {!isAdmin && (
            <div className="panel broker-playbook-panel">
              <div className="broker-playbook-head">
                <h3>Broker Playbook</h3>
                <span className={`broker-volatility-tag broker-volatility-tag-${volatilityProfile.tone}`}>
                  Avg range: {recentRangePct === null ? "—" : formatPercent(recentRangePct)}
                </span>
              </div>
              <div className="broker-playbook-grid">
                <article className="broker-playbook-card">
                  <p className="broker-card-title">Execution levels</p>
                  <div className="broker-card-rows">
                    <p>
                      Entry <b>{Number.isFinite(tradePlan?.entry) ? formatPrice(tradePlan.entry) : "—"}</b>
                    </p>
                    <p>
                      Target <b>{Number.isFinite(tradePlan?.target) ? formatPrice(tradePlan.target) : "—"}</b>
                    </p>
                    <p>
                      Stop <b>{Number.isFinite(tradePlan?.stop) ? formatPrice(tradePlan.stop) : "—"}</b>
                    </p>
                    <p>
                      Risk/Reward <b>{Number.isFinite(tradePlan?.riskReward) ? `${tradePlan.riskReward.toFixed(2)}R` : "—"}</b>
                    </p>
                  </div>
                </article>
                <article className="broker-playbook-card">
                  <p className="broker-card-title">Path structure</p>
                  <div className="broker-card-rows">
                    <p>
                      Up / down days{" "}
                      <b>
                        {forecastPathStats.upDays} / {forecastPathStats.downDays}
                      </b>
                    </p>
                    <p>
                      Best upside{" "}
                      <b className={forecastPathStats.pathHighPct !== null && forecastPathStats.pathHighPct >= 0 ? "text-up" : "text-down"}>
                        {formatSignedPercent(forecastPathStats.pathHighPct)}
                      </b>
                    </p>
                    <p>
                      Worst downside{" "}
                      <b className={forecastPathStats.pathLowPct !== null && forecastPathStats.pathLowPct <= 0 ? "text-down" : "text-up"}>
                        {formatSignedPercent(forecastPathStats.pathLowPct)}
                      </b>
                    </p>
                    <p>
                      Stop distance{" "}
                      <b>{Number.isFinite(tradePlan?.stopDistancePct) ? formatPercent(tradePlan.stopDistancePct) : "—"}</b>
                    </p>
                  </div>
                </article>
              </div>
            </div>
          )}

          <div className="panel chart-panel">
            <div className="chart-header">
              <h3>Market & Forecast</h3>
              <span>
                {symbol} · horizon {forecastDays}d
              </span>
            </div>
            <CandlestickChart
              candles={candles}
              forecastPath={forecastPath}
              forecastSpreadPath={forecastSpreadPath}
              selectedForecastDay={selectedForecastPoint?.day_ahead ?? null}
              onSelectForecastDay={setSelectedForecastDay}
              showTechnical={isAdmin}
              levelOverlays={brokerLevelOverlays}
            />
            <div className="legend">
              <span className="legend-item">
                <i className="dot history-dot" /> Historical candles
              </span>
              <span className="legend-item">
                <i className="dot forecast-dot" /> Ghost forecast candles
              </span>
              <span className="legend-item">
                <i className="dot spread-dot" /> Forecast spread band
              </span>
              {!isAdmin && brokerLevelOverlays.length > 0 && (
                <span className="legend-item">
                  <i className="dot levels-dot" /> Entry / target / stop
                </span>
              )}
            </div>
          </div>

          <div className="panel horizon-panel">
            <h3>Horizon Structure</h3>
            {forecastPath.length ? (
              <>
                <div className="horizon-bars horizon-bars-interactive">
                  {forecastPath.map((point, idx) => {
                    const isActive = selectedForecastPoint?.day_ahead === point.day_ahead;
                    const ratio = forecastPath.length > 1 ? idx / (forecastPath.length - 1) : 0;
                    const barHeight = Math.max(14, Math.round(36 - ratio * 18));

                    return (
                      <button
                        type="button"
                        key={`horizon-${point.day_ahead}`}
                        className={`horizon-bar-button ${isActive ? "is-active" : ""}`}
                        style={{ height: `${barHeight}px` }}
                        onClick={() => setSelectedForecastDay(point.day_ahead)}
                        onMouseEnter={() => setSelectedForecastDay(point.day_ahead)}
                      >
                        <span className="horizon-bar-label">D{point.day_ahead}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="horizon-detail">
                  <p className="horizon-detail-main">
                    Day {selectedForecastPoint?.day_ahead} ·{" "}
                    {formatDate(selectedForecastPoint?.predict_for_at)} ·{" "}
                    {formatPrice(selectedForecastPoint?.predicted_price)}
                  </p>
                  <p
                    className={`horizon-detail-sub ${
                      selectedForecastDelta === null
                        ? "muted"
                        : selectedForecastDelta >= 0
                          ? "text-up"
                          : "text-down"
                    }`}
                  >
                    {selectedForecastDelta === null
                      ? "Δ vs spot: —"
                      : `Δ vs spot: ${formatPrice(selectedForecastDelta)} (${selectedForecastDeltaPct?.toFixed(
                          2
                        )}%)`}
                  </p>
                </div>
              </>
            ) : (
              <p className="muted">Run prediction to unlock interactive horizon details.</p>
            )}
          </div>

          <div className="panel forecast-panel">
            <h3>Predicted Values</h3>
            {forecastPath.length ? (
              <>
                <div className="forecast-grid">
                  {forecastPath.map((point, idx) => {
                    const isActive = selectedForecastPoint?.day_ahead === point.day_ahead;
                    const previousReference =
                      idx === 0 ? currentPrice : forecastPath[idx - 1]?.predicted_price;
                    const stepDelta =
                      previousReference === null || previousReference === undefined
                        ? null
                        : point.predicted_price - previousReference;
                    const stepDeltaPct =
                      stepDelta !== null && previousReference
                        ? (stepDelta / previousReference) * 100
                        : null;

                    return (
                      <button
                        type="button"
                        className={`forecast-item ${isActive ? "is-active" : ""}`}
                        key={point.day_ahead}
                        onClick={() => setSelectedForecastDay(point.day_ahead)}
                        onMouseEnter={() => setSelectedForecastDay(point.day_ahead)}
                      >
                        <p className="forecast-day">Day {point.day_ahead}</p>
                        <p className="forecast-price">{formatPrice(point.predicted_price)}</p>
                        <p className="forecast-date">{formatDate(point.predict_for_at)}</p>
                        <p
                          className={`forecast-delta ${
                            stepDelta === null ? "muted" : stepDelta >= 0 ? "text-up" : "text-down"
                          }`}
                        >
                          {stepDelta === null
                            ? "Δ from prev: —"
                            : `Δ from prev: ${formatPrice(stepDelta)} (${stepDeltaPct?.toFixed(2)}%)`}
                        </p>
                      </button>
                    );
                  })}
                </div>
                <p className="muted forecast-hint">
                  Hover or click a day card to sync it with horizon and ghost candles.
                </p>
              </>
            ) : (
              <p className="muted">Run prediction to see daily forecast values.</p>
            )}
          </div>

          {isAdmin && (
            <div className="panel features-panel">
            <div className="features-header">
              <h3>Feature Explorer</h3>
              <span className="features-meta">
                {loadingFeatures && !featureItems.length
                  ? "Loading..."
                  : `${visibleFeatureItems.length}/${featureItems.length} features`}
              </span>
            </div>
            <div className="features-toolbar">
              <select
                className="control-select feature-filter-select"
                value={featureCategory}
                onChange={(event) => setFeatureCategory(event.target.value)}
              >
                <option value="all">all</option>
                {featureCategories.map((item) => (
                  <option key={`cat-${item.value}`} value={item.value}>
                    {item.value} ({item.count})
                  </option>
                ))}
              </select>
              <input
                className="control-input feature-search-input"
                type="text"
                placeholder="Search feature..."
                value={featureSearch}
                onChange={(event) => setFeatureSearch(event.target.value)}
              />
            </div>
            {loadingFeatures && !featureItems.length ? (
              <p className="muted">Preparing feature visualizations...</p>
            ) : visibleFeatureItems.length ? (
              <div className="feature-grid">
                {visibleFeatureItems.map((feature) => {
                  const rangeValue =
                    feature?.max_value !== null &&
                    feature?.max_value !== undefined &&
                    feature?.min_value !== null &&
                    feature?.min_value !== undefined
                      ? feature.max_value - feature.min_value
                      : null;
                  return (
                    <article
                      className={`feature-card ${selectedFeature?.name === feature.name ? "is-active" : ""}`}
                      key={feature.name}
                      role="button"
                      tabIndex={0}
                      onClick={() => setSelectedFeature(feature)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setSelectedFeature(feature);
                        }
                      }}
                    >
                      <div className="feature-card-head">
                        <p className="feature-name">{feature.name}</p>
                        <span className={`feature-badge feature-badge-${feature.category}`}>
                          {feature.category}
                        </span>
                      </div>
                      <FeatureSparkline
                        history={feature.history}
                        scaleHint={feature.scale_hint}
                        category={feature.category}
                      />
                      <div className="feature-stats">
                        <span>L: {formatFeatureValue(feature, feature.latest_value)}</span>
                        <span>μ: {formatFeatureValue(feature, feature.mean_value)}</span>
                        <span>Δ: {formatFeatureValue(feature, rangeValue)}</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <p className="muted">No features match current filter.</p>
            )}
            </div>
          )}
        </section>

        <aside className="sidebar-column">
          <div className={`panel confidence-panel ${isAdmin ? "" : "confidence-panel-broker"}`}>
            <p className="eyebrow eyebrow-with-hint">
              <span>{isAdmin ? "Model Confidence" : "Signal Confidence"}</span>
              {isAdmin && (
                <abbr className="stat-hint" title={confidenceHint}>
                  ⓘ
                </abbr>
              )}
            </p>
            <div className="confidence-main">
              <div className="confidence-value">
                {confidence}
                <span>%</span>
              </div>
              <p className="muted soft confidence-caption">
                {isAdmin
                  ? "Snapshot of the most important quality signals for the current model run."
                  : "Probability proxy derived from recent model error and current horizon."}
              </p>
            </div>
            {isAdmin ? (
              <div className="confidence-kpi-grid">
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">MAPE</span>
                  <b className="confidence-kpi-value">
                    {mapeValue === undefined || mapeValue === null ? "—" : formatPercent(mapeValue * 100)}
                  </b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">MAE</span>
                  <b className="confidence-kpi-value">
                    {formatMetricValue(maeValue)}
                  </b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">RMSE</span>
                  <b className="confidence-kpi-value">
                    {formatMetricValue(rmseValue)}
                  </b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">MSE</span>
                  <b className="confidence-kpi-value">
                    {formatMetricValue(mseValue)}
                  </b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Horizon</span>
                  <b className="confidence-kpi-value">{forecastDays}d</b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Freshness</span>
                  <b
                    className={`confidence-kpi-value ${
                      prediction?.request_data_is_fresh ? "confidence-kpi-up" : "confidence-kpi-down"
                    }`}
                  >
                    {prediction?.request_data_is_fresh ? "Fresh" : "Delayed"}
                  </b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Model</span>
                  <b className="confidence-kpi-value">{modelTypeLabel}</b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Train rows</span>
                  <b className="confidence-kpi-value">{prediction?.train_rows ?? "—"}</b>
                </div>
              </div>
            ) : (
              <div className="confidence-kpi-grid confidence-kpi-grid-broker">
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Signal</span>
                  <b className="confidence-kpi-value">{decisionSummary.action}</b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Move</span>
                  <b className="confidence-kpi-value">{decisionSummary.expectedMoveLabel}</b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Horizon</span>
                  <b className="confidence-kpi-value">{forecastDays}d</b>
                </div>
                <div className="confidence-kpi">
                  <span className="confidence-kpi-label">Data</span>
                  <b
                    className={`confidence-kpi-value ${
                      prediction?.request_data_is_fresh ? "confidence-kpi-up" : "confidence-kpi-down"
                    }`}
                  >
                    {prediction?.request_data_is_fresh ? "Fresh" : "Delayed"}
                  </b>
                </div>
              </div>
            )}
          </div>

          <div className="panel stats-panel">
            <h3>{isAdmin ? "Asset Metrics" : "Decision Snapshot"}</h3>
            {isAdmin ? (
              <>
                <div className="stat-row">
                  <StatLabel text="Last close" hint={fieldHints.lastClose} />
                  <b>{formatPrice(currentPrice)}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Forecast horizon" hint={fieldHints.forecastHorizon} />
                  <b>{forecastDays} days</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Train rows" hint={fieldHints.trainRows} />
                  <b>{prediction?.train_rows ?? "—"}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="MAE" hint={fieldHints.mae} />
                  <b>{formatMetricValue(maeValue)}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="RMSE" hint={fieldHints.rmse} />
                  <b>{formatMetricValue(rmseValue)}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="MAPE" hint={fieldHints.mape} />
                  <b>{mapeValue === undefined || mapeValue === null ? "—" : formatPercent(mapeValue * 100)}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="MSE" hint={fieldHints.mse} />
                  <b>{formatMetricValue(mseValue)}</b>
                </div>
              </>
            ) : (
              <>
                <div className="stat-row">
                  <span>Signal</span>
                  <b>{decisionSummary.action}</b>
                </div>
                <div className="stat-row">
                  <span>Last close</span>
                  <b>{formatPrice(currentPrice)}</b>
                </div>
                <div className="stat-row">
                  <span>Target day</span>
                  <b>
                    {selectedForecastPoint
                      ? `D${selectedForecastPoint.day_ahead} · ${formatDate(selectedForecastPoint.predict_for_at)}`
                      : "—"}
                  </b>
                </div>
                <div className="stat-row">
                  <span>Target price</span>
                  <b>{selectedForecastPoint ? formatPrice(selectedForecastPoint.predicted_price) : "—"}</b>
                </div>
                <div className="stat-row">
                  <span>Expected move</span>
                  <b>{decisionSummary.expectedMoveLabel}</b>
                </div>
              </>
            )}
          </div>

          {isAdmin && (
            <>
              <div className="panel model-panel">
                <h3>Model Context</h3>
                <div className="stat-row">
                  <StatLabel text="Model ID" hint={fieldHints.modelId} />
                  <b className="mono">{modelIdLabel}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Model type" hint={fieldHints.modelType} />
                  <b>{modelTypeLabel}</b>
                </div>
                {prediction?.ensemble_weights && (
                  <div className="stat-row">
                    <StatLabel text="Ensemble weights" hint="Вклад каждой компоненты ансамбля в итоговый прогноз." />
                    <b className="mono">
                      {Object.entries(prediction.ensemble_weights)
                        .filter(([, w]) => w > 0)
                        .map(([name, w]) => `${name}: ${Math.round(w * 100)}%`)
                        .join(" · ")}
                    </b>
                  </div>
                )}
                <div className="stat-row">
                  <StatLabel text="Loss function" hint={fieldHints.lossFunction} />
                  <b className="mono">{lossLabel}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Tuning scoring" hint={fieldHints.tuningScoring} />
                  <b className="mono">{scoringLabel}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Trained at" hint={fieldHints.trainedAt} />
                  <b>{trainedAtLabel}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Training data freshness" hint={fieldHints.trainingDataFreshness} />
                  <b className={prediction?.training_data_is_fresh ? "text-up" : "text-down"}>
                    {prediction?.training_data_is_fresh === undefined ||
                    prediction?.training_data_is_fresh === null
                      ? "—"
                      : prediction.training_data_is_fresh
                        ? "Fresh"
                        : "Delayed"}
                  </b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Training max lag" hint={fieldHints.trainingMaxLag} />
                  <b>{formatLagMinutes(prediction?.training_max_data_lag_minutes)}</b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Request data freshness" hint={fieldHints.requestDataFreshness} />
                  <b className={prediction?.request_data_is_fresh ? "text-up" : "text-down"}>
                    {prediction?.request_data_is_fresh ? "Fresh" : "Delayed"}
                  </b>
                </div>
                <div className="stat-row">
                  <StatLabel text="Request max lag" hint={fieldHints.requestMaxLag} />
                  <b>{formatLagMinutes(prediction?.request_max_data_lag_minutes)}</b>
                </div>
                <div className="window-section">
                  <p className="window-title">Training windows</p>
                  {trainingWindows.length ? (
                    <div className="window-list">
                      {trainingWindows.map((item) => (
                        <div className="window-item" key={`training-${item.symbol}`}>
                          <p className="window-symbol">{item.symbol}</p>
                          <p className="window-meta">{item.rows} rows</p>
                          <p className="window-meta">
                            {formatDate(item.start_at)} → {formatDate(item.end_at)}
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">No training window metadata.</p>
                  )}
                </div>
                <div className="window-section">
                  <p className="window-title">Request windows</p>
                  {requestWindows.length ? (
                    <div className="window-list">
                      {requestWindows.map((item) => (
                        <div className="window-item" key={`request-${item.symbol}`}>
                          <p className="window-symbol">{item.symbol}</p>
                          <p className="window-meta">{item.rows} rows</p>
                          <p className="window-meta">
                            {formatDate(item.start_at)} → {formatDate(item.end_at)}
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">No request window metadata.</p>
                  )}
                </div>
              </div>

              <div className="panel momentum-panel">
                <h3>Live Momentum</h3>
                <div className="momentum-list">
                  {liveRows.length ? (
                    liveRows.map((item, idx) => (
                      <div className="momentum-row" key={`${item.timestamp}-${idx}`}>
                        <span className={idx < 2 ? "text-down" : "text-up"}>
                          {Number(item.close).toLocaleString("en-US", { maximumFractionDigits: 2 })}
                        </span>
                        <span className="muted">
                          {Number(item.volume).toLocaleString("en-US", {
                            maximumFractionDigits: 2,
                          })}{" "}
                          vol
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="muted">No ticks yet</p>
                  )}
                </div>
              </div>
            </>
          )}
        </aside>
      </main>

      <footer className="mobile-footer">
        <button className={`btn btn-muted mobile-decision-btn mobile-decision-${decisionSummary.tone}`}>
          {decisionSummary.action}
        </button>
        <button className="btn btn-primary" onClick={runPredict} disabled={loadingPrediction}>
          Update
        </button>
      </footer>
      {selectedFeature && (
        <div
          className="feature-modal-backdrop"
          onClick={() => setSelectedFeature(null)}
          role="presentation"
        >
          <div
            className="feature-modal"
            role="dialog"
            aria-modal="true"
            aria-label={`Feature ${selectedFeature.name}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="feature-modal-header">
              <div>
                <p className="eyebrow">Feature detail</p>
                <h3 className="feature-modal-title">{selectedFeature.name}</h3>
              </div>
              <button
                className="btn btn-muted feature-modal-close"
                onClick={() => setSelectedFeature(null)}
                type="button"
              >
                Close
              </button>
            </div>
            <div className="feature-modal-badges">
              <span className={`feature-badge feature-badge-${selectedFeature.category}`}>
                {selectedFeature.category}
              </span>
              <span className="feature-badge feature-badge-other">{selectedFeature.scale_hint}</span>
            </div>
            <div className="feature-modal-stats">
              <span>Latest: {formatFeatureValue(selectedFeature, selectedFeature.latest_value)}</span>
              <span>Mean: {formatFeatureValue(selectedFeature, selectedFeature.mean_value)}</span>
              <span>Min: {formatFeatureValue(selectedFeature, selectedFeature.min_value)}</span>
              <span>Max: {formatFeatureValue(selectedFeature, selectedFeature.max_value)}</span>
              <span>Range: {formatFeatureValue(selectedFeature, selectedFeatureRangeValue)}</span>
              <span>Samples: {selectedFeature.history?.length ?? 0}</span>
            </div>
            <FeatureSparkline
              history={selectedFeature.history}
              scaleHint={selectedFeature.scale_hint}
              category={selectedFeature.category}
              detailed
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default TradingTerminalWidget;
