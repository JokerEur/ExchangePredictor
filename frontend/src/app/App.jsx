import TradingTerminalPage from "../pages/trading-terminal/ui/TradingTerminalPage";
import { useState } from "react";

const DEFAULT_ROLE = "broker";

function App() {
  const [role, setRole] = useState(DEFAULT_ROLE);
  return <TradingTerminalPage role={role} onRoleChange={setRole} />;
}

export default App;

