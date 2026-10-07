"""Signed application events with explicit canonicalization and external trust inputs."""

import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


DOMAIN = b"CuriousCoderAudit/v1\x00"


def canonical_bytes(value):
    def validate(item, depth=0):
        if depth > 20:
            raise ValueError("Audit payload nesting exceeds the protocol limit.")
        if item is None or isinstance(item, (str, bool)):
            return
        if type(item) is int and abs(item) <= 2 ** 53 - 1:
            return
        if isinstance(item, list):
            for child in item:
                validate(child, depth + 1)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child, depth + 1)
            return
        raise ValueError("Audit values must be bounded integers, strings, booleans, nulls, lists, or string-keyed objects; encode decimals explicitly.")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


class EventSigner:
    def __init__(self, private_key=None):
        self._key = private_key or Ed25519PrivateKey.generate()
        self.public_key = self._key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.key_id = hashlib.sha256(self.public_key).hexdigest()

    def sign_hash(self, event_hash):
        return self._key.sign(DOMAIN + bytes.fromhex(event_hash)).hex()


class AuditLedger:
    def __init__(self, connection, signer):
        self.connection, self.signer = connection, signer

    def append(self, event_key, actor, event_type, recorded_at, payload):
        if not self.connection.in_transaction:
            raise RuntimeError("Append must share the application's active transaction.")
        if any(not isinstance(value, str) or not value.strip() for value in (event_key, actor, event_type, recorded_at)):
            raise ValueError("Event identity, actor label, type, and recorded time are required.")
        previous = self.connection.execute("SELECT sequence, event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1").fetchone()
        sequence, previous_hash = (previous[0] + 1, previous[1]) if previous else (1, "0" * 64)
        envelope = {"protocol": 1, "sequence": sequence, "event_key": event_key, "actor": actor,
                    "event_type": event_type, "recorded_at": recorded_at, "payload": payload,
                    "previous_hash": previous_hash, "key_id": self.signer.key_id}
        encoded = canonical_bytes(envelope)
        event_hash = hashlib.sha256(encoded).hexdigest()
        signature = self.signer.sign_hash(event_hash)
        self.connection.execute("INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (sequence, event_key, encoded.decode(), previous_hash, event_hash, self.signer.key_id, signature))
        return {"sequence": sequence, "event_hash": event_hash, "key_id": self.signer.key_id}


def verify_events(events, trusted_public_keys, expected_head=None):
    previous = "0" * 64
    for expected_sequence, record in enumerate(events, 1):
        try:
            envelope = json.loads(record["envelope"])
            if record["sequence"] != expected_sequence or envelope["sequence"] != expected_sequence:
                return {"valid": False, "reason": "sequence mismatch"}
            if record["previous_hash"] != previous or envelope["previous_hash"] != previous:
                return {"valid": False, "reason": "broken hash linkage"}
            if envelope["key_id"] != record["key_id"] or envelope["event_key"] != record["event_key"] or envelope["protocol"] != 1:
                return {"valid": False, "reason": "envelope/header mismatch"}
            calculated = hashlib.sha256(canonical_bytes(envelope)).hexdigest()
            if calculated != record["event_hash"]:
                return {"valid": False, "reason": "payload digest mismatch"}
            public_key = bytes.fromhex(trusted_public_keys[record["key_id"]])
            if hashlib.sha256(public_key).hexdigest() != record["key_id"]:
                return {"valid": False, "reason": "key identity mismatch"}
            Ed25519PublicKey.from_public_bytes(public_key).verify(bytes.fromhex(record["signature"]), DOMAIN + bytes.fromhex(calculated))
            previous = calculated
        except (KeyError, TypeError, ValueError, InvalidSignature):
            return {"valid": False, "reason": "invalid encoding, untrusted key, or signature"}
    if expected_head is not None and previous != expected_head:
        return {"valid": False, "reason": "trusted head does not match; history may be incomplete"}
    return {"valid": True, "events": len(events), "head": previous,
            "scope": "Valid relative to supplied trusted keys/head; not proof of human identity, truthful input, or regulatory compliance."}
