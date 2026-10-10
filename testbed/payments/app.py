import random
import sys
import uuid

sys.path.insert(0, "/app/common")
from flask import Flask, jsonify, request
from logsetup import setup_logging
from faults import arm_fault, maybe_delay_or_fail, maybe_restart_process

app = Flask(__name__)
logger = setup_logging("payments")

logger.info("service started, version=1.0")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/charge")
def charge():
    request_id = request.args.get("request_id", str(uuid.uuid4())[:8])
    maybe_restart_process(logger)

    failed = maybe_delay_or_fail(logger, downstream_name=None)
    if failed:
        logger.error(f"payment authorization failed request_id={request_id}")
        return jsonify(error="payment failed", request_id=request_id), 503

    txn_id = str(uuid.uuid4())[:12]
    amount = round(random.uniform(5.0, 500.0), 2)
    logger.info(f"payment authorized txn_id={txn_id} amount={amount} request_id={request_id}")
    return jsonify(txn_id=txn_id, amount=amount, request_id=request_id)


@app.route("/_fault", methods=["POST"])
def inject_fault():
    body = request.get_json()
    arm_fault(body["type"], body["duration_s"])
    logger.info(f"fault armed type={body['type']} duration_s={body['duration_s']}")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
