import TradingTerminalWidget from "../../../widgets/trading-terminal/ui/TradingTerminalWidget";
function TradingTerminalPage({ role, onRoleChange }) {
  return <TradingTerminalWidget role={role} onRoleChange={onRoleChange} />;
  return <TradingTerminalWidget />;
}

export default TradingTerminalPage;

