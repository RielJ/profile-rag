import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./styles.css";

const root = document.getElementById("root");
if (root === null) {
  throw new Error('index.html is missing <div id="root">');
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
