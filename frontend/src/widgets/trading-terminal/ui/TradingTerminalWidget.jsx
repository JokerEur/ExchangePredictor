import { useEffect, useMemo, useRef, useState } from "react";
import { getCandles, getFeaturePreview, getPrediction } from "../../../shared/api";

const SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"];
const MODEL_OPTIONS = [
  { value: "xgboost", label: "XGBoost" },
  { value: "random_forest", label: "Random Forest" },
];
const MIN_FORECAST_DAYS = 1;
const MAX_FORECAST_DAYS = 30;
const LIVE_INTERVAL_OPTIONS = [5, 10, 15, 30, 60];
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
    "Тип алгоритма модели (например, XGBoost или Random Forest). Сравнивайте метрики между типами, чтобы выбрать лучший.",
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

function CandlestickChart({
  candles,
  forecastPath,
  selectedForecastDay,
  onSelectForecastDay,
}) {
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panRatio, setPanRatio] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const dragStateRef = useRef(null);
  const width = 980;
  const height = 320;
  const maxZoom = 8;
  const axis = {
    top: 18,
    right: 18,
    bottom: 42,
    left: 72,
  };
  useEffect(() => {
    setZoomLevel(1);
    setPanRatio(1);
    setHoveredIndex(null);
  }, [candles.length, forecastPath.length]);

  if (!candles.length) {
    return <div className="chart-empty">No market data</div>;
  }

  const recentRanges = candles
    .slice(-20)
    .map((item) => Math.max(Number(item.high) - Number(item.low), 0))
    .filter((range) => Number.isFinite(range) && range > 0);
  const baseRange =
    recentRanges.length > 0
      ? recentRanges.reduce((acc, item) => acc + item, 0) / recentRanges.length
      : Math.max(Number(candles[candles.length - 1].close) * 0.01, 1);
  const wickPadding = baseRange * 0.35;

  const forecastCandles = [];
  let previousClose = Number(candles[candles.length - 1].close);
  forecastPath.forEach((point, idx) => {
    const predictedClose = Number(point.predicted_price);
    const open = previousClose;
    const close = predictedClose;
    const high = Math.max(open, close) + wickPadding;
    const low = Math.max(0, Math.min(open, close) - wickPadding);
    forecastCandles.push({
      timestamp: point.predict_for_at,
      open,
      high,
      low,
      close,
      volume: null,
      type: "forecast",
      dayAhead: point.day_ahead ?? idx + 1,
    });
    previousClose = close;
  });

  const historicalCandles = candles.map((item) => ({
    ...item,
    type: "historical",
    dayAhead: null,
  }));
  const series = [...historicalCandles, ...forecastCandles];
  const minVisiblePoints = Math.min(
    series.length,
    Math.max(20, Math.min(70, forecastCandles.length + 14))
  );

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const resolveVisibleCount = (zoom) => {
    const raw = Math.round(series.length / Math.max(zoom, 1));
    return clamp(raw, minVisiblePoints, series.length);
  };

  const visibleCount = resolveVisibleCount(zoomLevel);
  const maxStart = Math.max(series.length - visibleCount, 0);
  const safePanRatio = maxStart > 0 ? clamp(panRatio, 0, 1) : 0;
  const startIndex = Math.round(maxStart * safePanRatio);
  const endIndex = startIndex + visibleCount;
  const visibleSeries = series.slice(startIndex, endIndex);
  const allHighValues = visibleSeries.map((item) => Number(item.high));
  const allLowValues = visibleSeries.map((item) => Number(item.low));
  const rawMin = Math.min(...allLowValues);
  const rawMax = Math.max(...allHighValues);
  const valuePadding = Math.max((rawMax - rawMin) * 0.08, rawMax * 0.002, 1);
  const min = rawMin - valuePadding;
  const max = rawMax + valuePadding;
  const span = Math.max(max - min, 1e-9);
  const totalPoints = Math.max(visibleSeries.length - 1, 1);
  const plotWidth = width - axis.left - axis.right;
  const plotHeight = height - axis.top - axis.bottom;
  const step = plotWidth / Math.max(totalPoints, 1);
  const candleWidth = Math.max(3, step * 0.62);
  const wickWidth = Math.max(1, candleWidth * 0.12);

  const toX = (idx) => axis.left + (idx / totalPoints) * plotWidth;
  const toY = (value) => axis.top + ((max - value) / span) * plotHeight;
  const forecastStartIndex = historicalCandles.length;
  const separatorX =
    forecastCandles.length > 0 &&
    forecastStartIndex > startIndex &&
    forecastStartIndex < endIndex
      ? toX(forecastStartIndex - startIndex) - step / 2
      : null;
  const yTickCount = 5;
  const yTicks = Array.from({ length: yTickCount }, (_, tickIndex) => {
    const ratio = tickIndex / Math.max(yTickCount - 1, 1);
    const value = max - ratio * span;
    return {
      value,
      y: toY(value),
    };
  });
  const xTickCount = Math.min(6, visibleSeries.length);
  const xTickIndexes = Array.from({ length: xTickCount }, (_, tickIndex) =>
    Math.round((tickIndex * Math.max(visibleSeries.length - 1, 0)) / Math.max(xTickCount - 1, 1))
  ).filter((value, index, items) => items.indexOf(value) === index);
  const xTicks = xTickIndexes.map((index) => ({
    index,
    x: toX(index),
    label: formatAxisDate(visibleSeries[index]?.timestamp),
    isForecast: visibleSeries[index]?.type === "forecast",
  }));

  const hoveredLocalIndex = hoveredIndex !== null ? hoveredIndex - startIndex : null;
  const hovered =
    hoveredLocalIndex !== null &&
    hoveredLocalIndex >= 0 &&
    hoveredLocalIndex < visibleSeries.length
      ? visibleSeries[hoveredLocalIndex]
      : null;
  const tooltipX = hovered ? toX(hoveredLocalIndex) : 0;
  const tooltipRightSide = tooltipX < width * 0.58;
  const tooltipBoxWidth = 210;
  const tooltipRows = hovered
    ? [
        hovered.type === "forecast"
          ? `Ghost candle · day ${hovered.dayAhead}`
          : "Historical candle",
        `Open: ${formatNumber(hovered.open)}`,
        `High: ${formatNumber(hovered.high)}`,
        `Low: ${formatNumber(hovered.low)}`,
        `Close: ${formatNumber(hovered.close)}`,
        `Volume: ${
          hovered.volume === null || hovered.volume === undefined
            ? "n/a (forecast)"
            : formatVolume(hovered.volume)
        }`,
      ]
    : [];
  const tooltipBoxHeight = 28 + tooltipRows.length * 16;
  const tooltipBoxX = tooltipRightSide
    ? tooltipX + 10
    : tooltipX - tooltipBoxWidth - 10;
  const tooltipBoxY = axis.top + 8;

  function formatNumber(value) {
    return Number(value).toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function formatVolume(value) {
    return Number(value).toLocaleString("en-US", {
      maximumFractionDigits: 2,
    });
  }

  function formatAxisPrice(value) {
    return Number(value).toLocaleString("en-US", {
      maximumFractionDigits: 0,
    });
  }

  function formatAxisDate(value) {
    if (!value) {
      return "—";
    }
    return new Date(value).toLocaleDateString("en-GB", {
      day: "2-digit",
      month: "short",
    });
  }

  function applyZoom(targetZoom, anchorRatio = 0.5) {
    const nextZoom = clamp(targetZoom, 1, maxZoom);
    const currentVisible = resolveVisibleCount(zoomLevel);
    const nextVisible = resolveVisibleCount(nextZoom);
    const currentMaxStart = Math.max(series.length - currentVisible, 0);
    const nextMaxStart = Math.max(series.length - nextVisible, 0);
    const currentStart = Math.round(currentMaxStart * safePanRatio);
    const anchor = clamp(anchorRatio, 0, 1);
    const anchorGlobalIndex = currentStart + anchor * Math.max(currentVisible - 1, 0);
    const nextStart = anchorGlobalIndex - anchor * Math.max(nextVisible - 1, 0);
    const clampedNextStart = clamp(nextStart, 0, nextMaxStart);
    setZoomLevel(nextZoom);
    setPanRatio(nextMaxStart > 0 ? clampedNextStart / nextMaxStart : 0);
  }

  function handleWheel(event) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const pointerRatio = clamp((event.clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
    const zoomDelta = event.deltaY < 0 ? 0.35 : -0.35;
    applyZoom(zoomLevel + zoomDelta, pointerRatio);
  }

  function handleMouseDown(event) {
    if (maxStart <= 0) {
      return;
    }
    dragStateRef.current = {
      clientX: event.clientX,
      panRatio: safePanRatio,
    };
    setIsDragging(true);
  }

  function handleMouseMove(event) {
    if (!dragStateRef.current) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const dx = event.clientX - dragStateRef.current.clientX;
    const baselineStart = dragStateRef.current.panRatio * maxStart;
    const shiftInPoints = (-dx / Math.max(rect.width, 1)) * Math.max(visibleCount, 1);
    const nextStart = clamp(baselineStart + shiftInPoints, 0, maxStart);
    setPanRatio(maxStart > 0 ? nextStart / maxStart : 0);
  }

  function finishDragging() {
    dragStateRef.current = null;
    setIsDragging(false);
  }

  function resetViewport() {
    setZoomLevel(1);
    setPanRatio(1);
    setHoveredIndex(null);
  }


  return (
    <div
      className={`chart-viewport ${isDragging ? "is-dragging" : ""}`}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={finishDragging}
      onMouseLeave={finishDragging}
    >
      <div
        className="chart-viewport-toolbar"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="chart-viewport-meta">
          <span>Zoom: {zoomLevel.toFixed(1)}x</span>
          <span>
            Window: {visibleSeries.length}/{series.length}
          </span>
          <span>Wheel = zoom · drag = move</span>
        </div>
        <div className="chart-viewport-actions">
          <button
            type="button"
            className="chart-viewport-btn"
            onClick={() => applyZoom(zoomLevel - 0.5)}
            disabled={zoomLevel <= 1}
          >
            −
          </button>
          <button
            type="button"
            className="chart-viewport-btn"
            onClick={() => applyZoom(zoomLevel + 0.5)}
            disabled={zoomLevel >= maxZoom}
          >
            +
          </button>
          <button
            type="button"
            className="chart-viewport-btn chart-viewport-btn-reset"
            onClick={resetViewport}
            disabled={zoomLevel === 1 && safePanRatio >= 0.999}
          >
            Reset
          </button>
        </div>
      </div>

      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {separatorX !== null && (
          <>
            <rect
              x={separatorX}
              y={axis.top}
              width={Math.max(width - axis.right - separatorX, 0)}
              height={plotHeight}
              fill="#f2e5bc"
              fillOpacity="0.6"
            />
            <line
              x1={separatorX}
              y1={axis.top}
              x2={separatorX}
              y2={height - axis.bottom}
              stroke="#7c6f64"
              strokeDasharray="4 4"
              strokeOpacity="0.65"
            />
          </>
        )}

        {yTicks.map((tick, idx) => (
          <g key={`y-tick-${idx}`}>
            <line
              x1={axis.left}
              y1={tick.y}
              x2={width - axis.right}
              y2={tick.y}
              stroke="#d5c4a1"
              strokeDasharray="3 4"
              strokeOpacity="0.75"
            />
            <text
              x={axis.left - 8}
              y={tick.y + 3}
              textAnchor="end"
              fill="#7c6f64"
              fontSize="10"
              fontWeight="600"
            >
              {formatAxisPrice(tick.value)}
            </text>
          </g>
        ))}

        <line
          x1={axis.left}
          y1={height - axis.bottom}
          x2={width - axis.right}
          y2={height - axis.bottom}
          stroke="#7c6f64"
          strokeOpacity="0.6"
        />

        {xTicks.map((tick) => (
          <g key={`x-tick-${tick.index}`}>
            <line
              x1={tick.x}
              y1={height - axis.bottom}
              x2={tick.x}
              y2={height - axis.bottom + 6}
              stroke="#7c6f64"
              strokeOpacity="0.6"
            />
            <text
              x={tick.x}
              y={height - 20}
              textAnchor="middle"
              fill={tick.isForecast ? "#076678" : "#7c6f64"}
              fontSize="10"
              fontWeight={tick.isForecast ? "700" : "600"}
            >
              {tick.label}
            </text>
          </g>
        ))}

        <text x={axis.left} y={axis.top - 6} fill="#7c6f64" fontSize="10" fontWeight="700">
          Price (USD)
        </text>
        <text
          x={axis.left + plotWidth / 2}
          y={height - 6}
          textAnchor="middle"
          fill="#7c6f64"
          fontSize="10"
          fontWeight="700"
        >
          Date
        </text>

        {visibleSeries.map((item, idx) => {
          const globalIdx = startIndex + idx;
          const x = toX(idx);
          const openY = toY(Number(item.open));
          const closeY = toY(Number(item.close));
          const highY = toY(Number(item.high));
          const lowY = toY(Number(item.low));
          const bodyY = Math.min(openY, closeY);
          const bodyHeight = Math.max(Math.abs(closeY - openY), 1.5);
          const isForecast = item.type === "forecast";
          const isSelectedForecast = isForecast && selectedForecastDay === item.dayAhead;
          const isHovered = hoveredIndex === globalIdx;
          const candleColor = Number(item.close) >= Number(item.open) ? "#98971a" : "#cc241d";

          return (
            <g
              key={`${item.type}-${item.timestamp}-${globalIdx}`}
              onMouseEnter={() => !isDragging && setHoveredIndex(globalIdx)}
              onMouseMove={() => !isDragging && setHoveredIndex(globalIdx)}
              onMouseLeave={() => !isDragging && setHoveredIndex(null)}
              onClick={() => {
                if (isForecast) {
                  onSelectForecastDay?.(item.dayAhead);
                }
              }}
              style={{ cursor: isForecast ? "pointer" : "default" }}
            >
              <line
                x1={x}
                y1={highY}
                x2={x}
                y2={lowY}
                stroke={candleColor}
                strokeWidth={wickWidth + (isSelectedForecast ? 0.9 : 0)}
                strokeOpacity={isForecast ? 0.55 : 0.9}
                strokeDasharray={isForecast ? "4 3" : undefined}
              />
              <rect
                x={x - candleWidth / 2}
                y={bodyY}
                width={candleWidth}
                height={bodyHeight}
                rx="1.5"
                fill={candleColor}
                fillOpacity={
                  isForecast
                    ? isSelectedForecast
                      ? 0.5
                      : isHovered
                        ? 0.38
                        : 0.25
                    : isHovered
                      ? 1
                      : 0.9
                }
                stroke={isForecast ? candleColor : "none"}
                strokeWidth={isForecast ? (isSelectedForecast ? 2.2 : 1.2) : 0}
                strokeDasharray={isForecast ? "4 3" : undefined}
              />
              <rect
                x={x - step / 2}
                y={axis.top}
                width={Math.max(step, 5)}
                height={plotHeight}
                fill="transparent"
              />
            </g>
          );
        })}

        {hovered && (
          <g>
            <line
              x1={tooltipX}
              y1={axis.top}
              x2={tooltipX}
              y2={height - axis.bottom}
              stroke="#7c6f64"
              strokeOpacity="0.45"
              strokeDasharray="3 3"
            />
            <rect
              x={tooltipBoxX}
              y={tooltipBoxY}
              width={tooltipBoxWidth}
              height={tooltipBoxHeight}
              rx="10"
              fill="#f9f5d7"
              stroke="#bdae93"
            />
            <text x={tooltipBoxX + 10} y={tooltipBoxY + 16} fill="#3c3836" fontSize="10" fontWeight="700">
              {new Date(hovered.timestamp).toLocaleDateString("en-GB", {
                day: "2-digit",
                month: "short",
                year: "numeric",
              })}
            </text>
            {tooltipRows.map((line, idx) => (
              <text
                key={`${line}-${idx}`}
                x={tooltipBoxX + 10}
                y={tooltipBoxY + 34 + idx * 16}
                fill={idx === 0 ? "#076678" : "#665c54"}
                fontSize="10"
                fontWeight={idx === 0 ? "700" : "500"}
              >
                {line}
              </text>
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}


function TradingTerminalWidget() {
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
          <p className="eyebrow">Neural Market Panel</p>
          <h1 className="title">Crypto Forecast Terminal</h1>
        </div>
        <div className="controls">
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
          <label className="live-toggle-wrap">
            <input
              type="checkbox"
              checked={liveEnabled}
              onChange={(event) => setLiveEnabled(event.target.checked)}
            />
            <span>Live</span>
          </label>
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
          <button className="btn btn-muted" onClick={refreshAllMarketData} disabled={loadingMarket || isLiveUpdating}>
            {loadingMarket ? "Refreshing..." : "Refresh"}
          </button>
          <button className="btn btn-primary" onClick={runPredict} disabled={loadingPrediction}>
            {loadingPrediction ? "Predicting..." : "Predict"}
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="layout-grid">
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
              <p className="fresh-label">Data freshness · 1m live</p>
              <p className={`fresh-state ${freshnessSource?.is_fresh ? "text-up" : "text-down"}`}>
                {freshnessSource?.is_fresh ? "Fresh" : "Delayed"}
              </p>
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
              <p className="fresh-meta">
                last live sync:{" "}
                {lastLiveUpdateAt ? new Date(lastLiveUpdateAt).toLocaleTimeString() : "—"}
              </p>
            </div>
          </div>

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
              selectedForecastDay={selectedForecastPoint?.day_ahead ?? null}
              onSelectForecastDay={setSelectedForecastDay}
            />
            <div className="legend">
              <span className="legend-item">
                <i className="dot history-dot" /> Historical candles
              </span>
              <span className="legend-item">
                <i className="dot forecast-dot" /> Ghost forecast candles
              </span>
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
        </section>

        <aside className="sidebar-column">
          <div className="panel confidence-panel">
            <p className="eyebrow eyebrow-with-hint">
              <span>Model Confidence</span>
              <abbr className="stat-hint" title={confidenceHint}>
                ⓘ
              </abbr>
            </p>
            <div className="confidence-main">
              <div className="confidence-value">
                {confidence}
                <span>%</span>
              </div>
              <p className="muted soft confidence-caption">
                Snapshot of the most important quality signals for the current model run.
              </p>
            </div>
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
          </div>

          <div className="panel stats-panel">
            <h3>Asset Metrics</h3>
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
          </div>

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
        </aside>
      </main>

      <footer className="mobile-footer">
        <button className="btn btn-muted">Buy</button>
        <button className="btn btn-primary" onClick={runPredict} disabled={loadingPrediction}>
          Predict
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
