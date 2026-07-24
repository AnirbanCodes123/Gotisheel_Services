/**
 * Vite/React entry — production UI currently ships from /frontend/ui
 * (no build step). Build this app with `npm run build` to replace ui via dist/.
 */
export default function App() {
  return (
    <div style={{ fontFamily: "DM Sans, sans-serif", padding: 24, background: "#0b0f14", color: "#e8eef6", minHeight: "100vh" }}>
      <h1>Gotisheel AI 2.0</h1>
      <p>React scaffold ready. The live dashboard is served from <code>frontend/ui</code> by FastAPI.</p>
      <p>
        Run the API and open <a href="http://localhost:9100" style={{ color: "#3dd6c6" }}>http://localhost:9100</a>
      </p>
    </div>
  );
}
