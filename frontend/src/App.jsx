import React, { useState } from "react";
import { HeartPulse } from "lucide-react";
import Prediction from "./pages/Prediction";
import Comparison from "./pages/Comparison";

export default function App() {
  const [page, setPage] = useState("prediction");
  return (
    <div className="site">
      <header className="site-header">
        <div className="header-inner">
          <a
            className="brand"
            href="#prediction"
            onClick={(event) => {
              event.preventDefault();
              setPage("prediction");
            }}
            aria-label="Cardiofy home"
          >
            <span className="brand-symbol">
              <HeartPulse size={23} strokeWidth={1.7} />
            </span>
            cardiofy<span className="brand-period">.</span>
          </a>
          <nav aria-label="Main navigation">
            <button
              className={page === "prediction" ? "active" : ""}
              aria-current={page === "prediction" ? "page" : undefined}
              onClick={() => setPage("prediction")}
            >
              Prediction
            </button>
            <button
              className={page === "comparison" ? "active" : ""}
              aria-current={page === "comparison" ? "page" : undefined}
              onClick={() => setPage("comparison")}
            >
              Model comparison
            </button>
          </nav>
          <span className="header-note">A heart health research project</span>
        </div>
      </header>
      <main className="site-main">
        {page === "prediction" ? <Prediction /> : <Comparison />}
      </main>
      <footer className="site-footer">
        <a
          className="footer-brand"
          href="#prediction"
          onClick={(event) => {
            event.preventDefault();
            setPage("prediction");
          }}
        >
          cardiofy.
        </a>
        <p>Experimental research only. Not a medical diagnosis.</p>
        <span>Local research prototype</span>
      </footer>
    </div>
  );
}
