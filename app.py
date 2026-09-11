from pathlib import Path
from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
from career import train

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "career_model.joblib"
app = Flask(__name__)

CAREER_INFO = {
    "Software Developer": {"summary":"Builds software applications, APIs and backend or frontend systems.","skills":["Python/Java","DSA","Git","SQL","APIs"],"roadmap":["Strengthen programming","Practice DSA","Build 2 projects","Learn Git + SQL","Apply for internships"]},
    "Data Analyst": {"summary":"Turns business or product data into useful insights and reports.","skills":["Python","Pandas","SQL","Statistics","Power BI/Tableau"],"roadmap":["Learn Excel","Master SQL","Practice Pandas","Learn visualization","Build an analytics portfolio"]},
    "Web Developer": {"summary":"Creates responsive websites and web applications.","skills":["HTML/CSS","JavaScript","React","APIs","Git"],"roadmap":["Master HTML/CSS","Learn JavaScript","Build responsive sites","Learn React","Deploy a full-stack project"]},
    "UI/UX Designer": {"summary":"Designs usable digital experiences, interfaces and user flows.","skills":["Figma","Wireframing","Typography","User research","Prototyping"],"roadmap":["Learn design principles","Practice Figma","Create wireframes","Build 3 case studies","Publish a portfolio"]},
    "Software Tester": {"summary":"Checks software quality through manual and automated testing.","skills":["Testing basics","Test cases","SQL","API testing","Selenium/Playwright"],"roadmap":["Learn SDLC","Write test cases","Learn API testing","Automate tests","Build a testing project"]},
    "Business Analyst": {"summary":"Connects business needs with data, processes and technology teams.","skills":["Excel","SQL","Requirements","Communication","Data visualization"],"roadmap":["Learn requirements gathering","Improve Excel","Learn SQL","Practice dashboards","Write business case studies"]},
    "Technical Support": {"summary":"Helps users solve technical issues and keeps systems working reliably.","skills":["Troubleshooting","Networking","Linux","Communication","Ticketing tools"],"roadmap":["Learn computer fundamentals","Study networking","Practice Linux","Learn troubleshooting","Practice support scenarios"]},
    "Technical Writer": {"summary":"Creates clear technical documentation, guides and developer content.","skills":["Writing","Documentation","Markdown","Research","Basic programming"],"roadmap":["Improve technical writing","Learn Markdown","Document a project","Learn API documentation","Build a documentation portfolio"]},
}

def get_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

def predict_scores(values):
    artifact = get_model()
    row = np.array([[float(values[name]) for name in artifact["features"]]])
    probabilities = artifact["model"].predict_proba(row)[0]
    ranked = sorted(zip(artifact["label_encoder"].classes_, probabilities), key=lambda x:x[1], reverse=True)
    return [{"career":career,"score":round(float(score)*100,1)} for career,score in ranked[:5]]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        artifact = get_model()
        values = {}
        for name in artifact["features"]:
            value = payload.get(name)
            if value is None or value == "":
                return jsonify({"error":f"Missing value: {name}"}), 400
            values[name] = float(value)
        ranking = predict_scores(values)
        top = ranking[0]
        info = CAREER_INFO.get(top["career"], {"summary":"A career path matched to your profile.","skills":[],"roadmap":[]})
        return jsonify({"recommendation":top,"ranking":ranking,"career_info":info})
    except ValueError:
        return jsonify({"error":"All assessment values must be numeric."}), 400
    except Exception as exc:
        return jsonify({"error":str(exc)}), 500

@app.route("/api/careers")
def careers():
    return jsonify(CAREER_INFO)

if __name__ == "__main__":
    app.run(debug=True)
