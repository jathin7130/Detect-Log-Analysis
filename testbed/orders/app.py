import sys
import uuid

sys.path.insert(0, "/app/common")
import requests
from flask import Flask, jsonify, request
from logsetup import setup_logging
from faults import arm_fault, maybe_delay_or_fail, maybe_restart_process

app = Flask(__name__)
logger = setup_logging("orders")

logger.info("service started, version=1.0")

INVENTORY_URL = "http://inventory:5000"
PAYMENTS_URL = "http://payments:5000"


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/place")
def place_order():
    request_id = request.args.get("request_id", str(uuid.uuid4())[:8])
    maybe_restart_process(logger)

    failed = maybe_delay_or_fail(logger, downstream_name="inventory")
    if failed:
        logger.error(f"order placement aborted, upstream degradation request_id={request_id}")
        return jsonify(error="order failed", request_id=request_id), 503

    logger.info(f"checking inventory for order request_id={request_id}")
    try:
        inv_resp = requests.get(f"{INVENTORY_URL}/check", params={"request_id": request_id}, timeout=5)
        if inv_resp.status_code != 200:
            logger.error(f"inventory service returned {inv_resp.status_code} request_id={request_id}")
            return jsonify(error="inventory unavailable", request_id=request_id), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"inventory service call failed: {type(e).__name__} request_id={request_id}")
        return jsonify(error="inventory unreachable", request_id=request_id), 502

    logger.info(f"charging payment for order request_id={request_id}")
    try:
        pay_resp = requests.get(f"{PAYMENTS_URL}/charge", params={"request_id": request_id}, timeout=5)
        if pay_resp.status_code != 200:
            logger.error(f"payments service returned {pay_resp.status_code} request_id={request_id}")
            return jsonify(error="payment unavailable", request_id=request_id), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"payments service call failed: {type(e).__name__} request_id={request_id}")
        return jsonify(error="payment unreachable", request_id=request_id), 502

    order_id = str(uuid.uuid4())[:10]
    logger.info(f"order placed successfully order_id={order_id} request_id={request_id}")
    return jsonify(order_id=order_id, request_id=request_id)


@app.route("/_fault", methods=["POST"])
def inject_fault():
    body = request.get_json()
    arm_fault(body["type"], body["duration_s"])
    logger.info(f"fault armed type={body['type']} duration_s={body['duration_s']}")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
