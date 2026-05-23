from flask import Flask, render_template, request

import logging

import os

import pickle



# NOTE: No news API, verification, or plotly imports at module level — keeps startup fast.





def _configure_logging() -> None:

    level = logging.DEBUG if os.environ.get("VERITAS_DEBUG") else logging.INFO

    if not logging.getLogger().handlers:

        logging.basicConfig(

            level=level,

            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",

            datefmt="%H:%M:%S",

        )





_configure_logging()



app = Flask(__name__, template_folder='templates', static_folder='static')



def _ensure_ml_artifacts():

    """Shared ML cache — single disk read per process."""

    from model_cache import get_models

    return get_models()







def fake_news_det(news: str):

    """Predict FAKE/REAL for a single news text.



    Fix: ensure we pass a 2D TF-IDF matrix into the scikit-learn model.

    """

    loaded_model, _ = _ensure_ml_artifacts()

    try:

        # Preferred: model is already a Pipeline(vectorizer -> classifier)

        # and can accept raw text.

        try:

            pred = loaded_model.predict([news])

            if pred is not None:

                label = pred[0]

                return "Fake ⚠️" if int(label) == 0 else "Real ✅"

        except Exception:

            pass



        # Fallback: load vectorizer separately (if present)

        vectorizer = None

        for vf in ("vectorizer.pkl", "tfidf_vectorizer.pkl", "tfidf.pkl"):

            try:

                with open(vf, "rb") as f:

                    vectorizer = pickle.load(f)

                    break

            except Exception:

                continue



        # If still missing, try extracting from a Pipeline

        if vectorizer is None and hasattr(loaded_model, "named_steps"):

            for k in ("tfidf", "vectorizer", "tfidf_vectorizer"):

                if k in loaded_model.named_steps:

                    vectorizer = loaded_model.named_steps[k]

                    break



        if vectorizer is None:

            raise RuntimeError("TF-IDF vectorizer not found and model did not accept raw text")



        # REQUIRED: transform to 2D

        X = vectorizer.transform([news])

        if getattr(X, "ndim", 2) == 1:

            X = X.reshape(1, -1)



        pred = loaded_model.predict(X)

        label = pred[0]

        return "Fake ⚠️" if int(label) == 0 else "Real ✅"



    except Exception as e:

        # Proper exception handling for the UI layer

        raise RuntimeError(str(e))







@app.route('/')
def home():

    # Server-rendered analytics so dashboard cards are never blank
    try:
        from analytics import build_analytics_figures
        analytics = build_analytics_figures()
    except Exception:
        analytics = None

    return render_template('index.html', analytics=analytics)






@app.route('/history')
def history_endpoint():
    """Return persisted prediction history as JSON."""
    from flask import jsonify
    from history import load_history

    rows = load_history(limit=200)
    return jsonify({"count": len(rows), "history": rows})


@app.route('/predict', methods=['POST'])
def predict():


    message = request.form.get('news', '')

    if not message.strip():

        return render_template(

            'index.html',

            prediction='Please enter news text.',

            news_input=message,

        )



    try:

        # Live verification runs only here — never at startup

        from verification import run_full_verification

        result = run_full_verification(message)



        # Log history event (never break UI)

        try:

            from history import log_verification, build_history_event



            ml_score = result.get('ml_confidence')
            ml_pred_display = result.get('ml_display')
            hybrid_pred = result.get('prediction')

            trusted_count = int(result.get('trusted_sources_count') or 0)
            verified_sources_count = result.get('verified_sources_count')
            if verified_sources_count is None:
                verified_sources_count = trusted_count
            verified_sources_count = int(verified_sources_count or 0)

            live_verification_score = result.get('live_score')
            source_trust_score = result.get('credibility_score')

            api_used = result.get('api_used')
            provider_used = result.get('provider_used') or api_used

            vr = result.get('verification_result') or {
                'match_count': result.get('match_count'),
                'trusted_count': result.get('trusted_count'),
                'warning': result.get('warning'),
            }

            event = build_history_event(
                query=message,
                ml_prediction=ml_pred_display,
                hybrid_prediction=hybrid_pred,
                final_confidence=float(result.get('final_confidence') or 0.0),
                ml_score=ml_score,
                live_verification_score=live_verification_score,
                source_trust_score=source_trust_score,
                provider_used=provider_used,
                trusted_sources_count=trusted_count,
                verified_sources_count=verified_sources_count,
                api_used=api_used,
                verification_result=vr,
            )

            log_verification(event)
            print('[HISTORY] prediction saved')


        except Exception:

            pass



        # Build analytics after each prediction (server-rendered Plotly HTML)
        try:
            from analytics import build_analytics_figures
            analytics = build_analytics_figures()
            print('[ANALYTICS] charts updated')
        except Exception:
            analytics = None

        return render_template(
            'index.html',
            result=result,
            prediction=result['prediction'],
            news_input=result.get('news_input', message),
            analytics=analytics,
        )


    except Exception as e:

        try:

            prediction = fake_news_det(message)

            return render_template(

                'index.html',

                prediction=prediction,

                news_input=message,

                warning=f"Hybrid verification unavailable ({e}). Showing ML-only result.",

            )

        except Exception as ml_err:

            return render_template(

                'index.html',

                prediction=f"Something went wrong: {ml_err}",

                news_input=message,

            )





if __name__ == '__main__':

    # Load .env only when starting server (not on werkzeug reloader parent quirks)

    try:

        from api_config import ensure_env_loaded, log_startup_status

        ensure_env_loaded()

        log_startup_status()

    except Exception:

        pass



    logging.getLogger("veritas.app").info("Starting Flask — live APIs run on /predict only")

    try:

        from model_cache import preload_models_async, preload_models_blocking, is_loaded

        preload_models_async()

        if not is_loaded():

            preload_models_blocking()

    except Exception:

        pass

    # use_reloader=False avoids double import hang in debug mode

    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)


