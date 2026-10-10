import sys
import uuid
import random

sys.path.insert(0, "/app/common")
import requests
from flask import Flask, jsonify, request
from logsetup import setup_logging
from faults import arm_fault, maybe_delay_or_fail, maybe_restart_process

app = Flask(__name__)
logger = setup_logging("gateway")

logger.info("service started, version=1.0")

ORDERS_URL = "http://orders:5000"


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/checkout")
def checkout():
    request_id = str(uuid.uuid4())[:8]
    client_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
    logger.info(f"received checkout request request_id={request_id} client_ip={client_ip}")
    maybe_restart_process(logger)

    failed = maybe_delay_or_fail(logger, downstream_name="orders")
    if failed:
        logger.error(f"checkout aborted before dispatch request_id={request_id}")
        return jsonify(error="checkout failed", request_id=request_id), 503

    try:
        resp = requests.get(f"{ORDERS_URL}/place", params={"request_id": request_id}, timeout=8)
        if resp.status_code != 200:
            logger.error(f"orders service returned {resp.status_code} request_id={request_id}")
            return jsonify(error="order failed", request_id=request_id), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"orders service call failed: {type(e).__name__} request_id={request_id}")
        return jsonify(error="orders unreachable", request_id=request_id), 502

    logger.info(f"checkout completed successfully request_id={request_id}")
    return jsonify(status="ok", request_id=request_id, order=resp.json())


@app.route("/_fault", methods=["POST"])
def inject_fault():
    body = request.get_json()
    arm_fault(body["type"], body["duration_s"])
    logger.info(f"fault armed type={body['type']} duration_s={body['duration_s']}")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
