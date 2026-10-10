"""PII masking for raw log content.

Masks IP addresses, file paths, and credential-like strings before any
template mining or modeling touches the text, per the project's ethics plan
(Section 3.4 of the CA2 proposal): public Loghub data is treated as
potentially sensitive even though it's publicly accessible.
"""
import re

IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
PATH_RE = re.compile(r"(?:/[\w\-.$]+){2,}")
CRED_RE = re.compile(
    r"\b(?:password|passwd|token|secret|api[_-]?key|auth)\s*[=:]\s*\S+",
    re.IGNORECASE,
)
# BGL-style node identifiers, e.g. R02-M1-N0-C:J12-U11
NODE_RE = re.compile(r"\b[A-Z]\d{2}-[A-Z]\d-[A-Z]\d(?:-[A-Z])?(?::[A-Z]\d{2}-[A-Z]\d{2})?\b")
USER_RE = re.compile(r"\buser[=:]\s*\S+", re.IGNORECASE)
# Thunderbird-style "user@host" locations, e.g. src@aadmin1, local@tbird-admin1
USERHOST_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\b")
# hostnames embedded in free text, e.g. tbird-admin1, aadmin1, node-12
HOSTNAME_RE = re.compile(r"\b(?:tbird-admin|aadmin|admin)\d+\b", re.IGNORECASE)


def mask_content(text: str) -> str:
    text = IPV4_RE.sub("<IP>", text)
    text = CRED_RE.sub("<CRED>", text)
    text = USER_RE.sub("user=<USER>", text)
    text = USERHOST_RE.sub("<USER>@<HOST>", text)
    text = NODE_RE.sub("<NODE>", text)
    text = HOSTNAME_RE.sub("<HOST>", text)
    text = PATH_RE.sub("<PATH>", text)
    return text
