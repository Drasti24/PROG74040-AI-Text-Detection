import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeText = async () => {
    if (!text.trim()) {
      setError("Please enter some text first.");
      setResult(null);
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const response = await axios.post(
        "https://prog74040-ai-text-detection.onrender.com/predict",
        {
          text: text
        }
      );
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(
        "Could not analyze the text. Please make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearText = () => {
    setText("");
    setResult(null);
    setError("");
  };

  return (
    <div className="page">
      <div className="container">
        <div className="header">
          <h1>AI Text Detector</h1>
          <p>
            Analyze text using our fine-tuned SVM model.
          </p>
        </div>

        <div className="card">
          <label htmlFor="text-input">
            Enter text
          </label>

          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste or type text here..."
            rows="12"
          />

          <div className="info-row">
            <span>
              {text.length} characters
            </span>
          </div>

          <div className="button-row">
            <button
              className="secondary-button"
              onClick={clearText}
            >
              Clear
            </button>

            <button
              className="primary-button"
              onClick={analyzeText}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Text"}
            </button>
          </div>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

          {result && (
            <div className="result-card">
              <h2>Result</h2>

              <div className="prediction">
                {result.prediction === "AI Generated"
                  ? "Likely AI-Generated"
                  : "Likely Human-Written"}
              </div>

              <div className="confidence">
                Model confidence score:{" "}
                {(result.confidence * 100).toFixed(2)}%
              </div>

              <div className="disclaimer">
                This score reflects the model&apos;s confidence, not proof of authorship.
                Results may be less reliable for writing styles or sources that differ
                from the training data.
              </div>

              {result && (
                <div className="result-section">
                  <h3>Result</h3>

                  <h2>
                    {result.prediction === "AI Generated"
                      ? "Likely AI-Generated"
                      : "Likely Human-Written"}
                  </h2>

                  <p>
                    Model confidence score:{" "}
                    {(result.confidence * 100).toFixed(2)}%
                  </p>

                  <div className="disclaimer">
                    This score reflects the model's confidence, not proof of authorship.
                    Results may be less reliable for writing styles or sources that differ
                    from the training data.
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="footer-note">
          Long inputs may be truncated during analysis.
        </div>
      </div>
    </div>
  );
}

export default App;