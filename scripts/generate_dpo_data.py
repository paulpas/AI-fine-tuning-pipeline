#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Data Generator

Generates chosen/rejected pairs for preference learning.
This teaches the model:
- Good coding practices vs bad practices
- Secure code vs insecure code
- Efficient code vs inefficient code
- Helpful responses vs unhelpful responses

DPO is more effective than standard instruction tuning for
teaching nuanced preferences.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class DPOPair:
    """A preference pair for DPO training"""
    prompt: str
    chosen: str
    rejected: str
    category: str
    severity: str = "medium"  # low, medium, high


class DPODataGenerator:
    """Generate DPO training pairs from anti-patterns and best practices"""

    def __init__(self):
        self.pairs = []
        self._load_patterns()

    def _load_patterns(self):
        """Load all anti-pattern -> best-practice pairs"""

        # ============ PYTHON ANTI-PATTERNS ============
        self.python_patterns = [
            # Exception handling
            {
                "prompt": "How do I handle exceptions in Python?",
                "rejected": """Just catch everything to be safe:

```python
try:
    do_something()
except:
    pass
```

This catches all errors so your program won't crash.""",
                "chosen": """Use specific exception handling:

```python
try:
    do_something()
except FileNotFoundError:
    logger.warning("File not found, using defaults")
    use_defaults()
except PermissionError as e:
    logger.error(f"Permission denied: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

**Why this is better:**
- Catches specific exceptions you can handle
- Logs errors for debugging
- Re-raises unexpected errors (don't hide bugs)
- Never use bare `except:` or `except Exception: pass`""",
                "category": "exception_handling",
                "severity": "high",
            },

            # Resource management
            {
                "prompt": "How do I read a file in Python?",
                "rejected": """Here's how to read a file:

```python
f = open('data.txt')
content = f.read()
f.close()
```""",
                "chosen": """Use a context manager to ensure the file is properly closed:

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()
```

**Why this is better:**
- File is automatically closed, even if an exception occurs
- Explicit encoding prevents platform-specific issues
- No risk of forgetting to close the file
- Cleaner, more Pythonic code""",
                "category": "resource_management",
                "severity": "medium",
            },

            # Mutable default arguments
            {
                "prompt": "How do I write a function with default list argument?",
                "rejected": """Here's a function with a default list:

```python
def add_item(item, items=[]):
    items.append(item)
    return items
```

Just use an empty list as the default.""",
                "chosen": """Use `None` as the default and create the list inside:

```python
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

**Why this is important:**
- Default mutable arguments are shared between ALL calls
- The `[]` is created once when the function is defined, not each call
- This causes subtle bugs where items accumulate unexpectedly

```python
# With the bad version:
add_item(1)  # Returns [1]
add_item(2)  # Returns [1, 2] - BUG! Expected [2]
```""",
                "category": "mutable_defaults",
                "severity": "high",
            },

            # String concatenation
            {
                "prompt": "How do I build a string from a list of items?",
                "rejected": """Loop and concatenate:

```python
result = ""
for item in items:
    result += str(item) + ", "
result = result[:-2]
```""",
                "chosen": """Use `join()` for efficient string building:

```python
result = ", ".join(str(item) for item in items)
```

**Why this is better:**
- O(n) vs O(n²) complexity - much faster for large lists
- String concatenation creates new strings each iteration
- `join()` is optimized to allocate memory once
- More readable and Pythonic""",
                "category": "performance",
                "severity": "medium",
            },

            # Type checking
            {
                "prompt": "How do I check if something is a list?",
                "rejected": """Use type():

```python
if type(x) == list:
    process_list(x)
```""",
                "chosen": """Use `isinstance()` for type checking:

```python
if isinstance(x, list):
    process_list(x)

# Or for multiple types:
if isinstance(x, (list, tuple)):
    process_sequence(x)
```

**Why this is better:**
- Works with subclasses (a custom list type would fail `type() == list`)
- More Pythonic - follows Liskov substitution principle
- Can check multiple types at once
- Better for duck typing philosophy""",
                "category": "type_checking",
                "severity": "low",
            },
        ]

        # ============ KUBERNETES ANTI-PATTERNS ============
        self.kubernetes_patterns = [
            # Client reuse
            {
                "prompt": "How do I list pods in Kubernetes with Python?",
                "rejected": """Here's how to list pods:

```python
from kubernetes import client, config

def list_pods():
    config.load_kube_config()
    return client.CoreV1Api().list_pod_for_all_namespaces()

# Call multiple times
pods1 = list_pods()
pods2 = list_pods()
```""",
                "chosen": """Create the client once and reuse it:

```python
from kubernetes import client, config

# Initialize once
config.load_kube_config()
v1 = client.CoreV1Api()

def list_pods(namespace: str = None):
    if namespace:
        return v1.list_namespaced_pod(namespace)
    return v1.list_pod_for_all_namespaces()

# Reuse the same client
pods1 = list_pods()
pods2 = list_pods("kube-system")
```

**Why this is better:**
- Avoids re-reading kubeconfig on every call
- Reuses HTTP connections (connection pooling)
- Much faster for multiple operations
- Lower memory usage""",
                "category": "resource_reuse",
                "severity": "medium",
            },

            # Polling vs Watch
            {
                "prompt": "How do I monitor pods in real-time with Kubernetes Python client?",
                "rejected": """Poll in a loop:

```python
import time
from kubernetes import client, config

config.load_kube_config()
v1 = client.CoreV1Api()

while True:
    pods = v1.list_pod_for_all_namespaces()
    for pod in pods.items:
        print(f"{pod.metadata.name}: {pod.status.phase}")
    time.sleep(5)
```""",
                "chosen": """Use the Watch API for efficient real-time updates:

```python
from kubernetes import client, config, watch

config.load_kube_config()
v1 = client.CoreV1Api()
w = watch.Watch()

for event in w.stream(v1.list_pod_for_all_namespaces, timeout_seconds=0):
    pod = event['object']
    event_type = event['type']  # ADDED, MODIFIED, DELETED

    print(f"{event_type}: {pod.metadata.name} - {pod.status.phase}")

    # Can filter and stop when needed
    if pod.metadata.name == "target-pod" and pod.status.phase == "Running":
        w.stop()
```

**Why Watch is better:**
- Instant updates (no polling delay)
- Much lower API server load
- More efficient (uses HTTP streaming)
- Only sends changes, not full list each time
- Built-in reconnection handling""",
                "category": "api_efficiency",
                "severity": "high",
            },

            # Error handling
            {
                "prompt": "How do I handle errors when creating Kubernetes resources?",
                "rejected": """Just create it:

```python
from kubernetes import client

def create_configmap(name, data):
    v1 = client.CoreV1Api()
    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name),
        data=data
    )
    return v1.create_namespaced_config_map("default", cm)
```""",
                "chosen": """Handle specific API exceptions:

```python
from kubernetes import client
from kubernetes.client.rest import ApiException

def create_configmap(name: str, data: dict, namespace: str = "default"):
    v1 = client.CoreV1Api()

    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=name),
        data=data
    )

    try:
        return v1.create_namespaced_config_map(namespace, cm)
    except ApiException as e:
        if e.status == 409:
            # Already exists - update instead
            return v1.patch_namespaced_config_map(name, namespace, cm)
        elif e.status == 403:
            raise PermissionError(f"No permission to create ConfigMap: {e.reason}")
        elif e.status == 422:
            raise ValueError(f"Invalid ConfigMap spec: {e.body}")
        else:
            raise RuntimeError(f"K8s API error ({e.status}): {e.reason}") from e
```

**Why this is better:**
- Handles common cases (already exists, no permission)
- Provides meaningful error messages
- Makes idempotent (safe to run multiple times)
- Converts K8s errors to Python exceptions for easier handling""",
                "category": "error_handling",
                "severity": "high",
            },
        ]

        # ============ SECURITY ANTI-PATTERNS ============
        self.security_patterns = [
            # Hardcoded secrets
            {
                "prompt": "How do I connect to a database in Python?",
                "rejected": """Here's how to connect:

```python
import psycopg2

conn = psycopg2.connect(
    host="db.example.com",
    database="myapp",
    user="admin",
    password="super_secret_123"
)
```""",
                "chosen": """Never hardcode credentials. Use environment variables:

```python
import os
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

# Or use a connection URL
conn = psycopg2.connect(os.environ["DATABASE_URL"])
```

**For production, use a secret manager:**
```python
from google.cloud import secretmanager

def get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/my-project/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

password = get_secret("db-password")
```

**Why this matters:**
- Hardcoded secrets get committed to version control
- They appear in logs, stack traces, and error messages
- Anyone with code access has credential access
- Rotating credentials requires code changes""",
                "category": "secrets",
                "severity": "critical",
            },

            # SQL injection
            {
                "prompt": "How do I query a database with user input in Python?",
                "rejected": """Use f-strings to build the query:

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
```""",
                "chosen": """Always use parameterized queries:

```python
def get_user(user_id: int):
    # Parameterized query - safe from injection
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

def search_users(name: str):
    # Even for LIKE queries, use parameters
    query = "SELECT * FROM users WHERE name LIKE %s"
    cursor.execute(query, (f"%{name}%",))
    return cursor.fetchall()
```

**Why parameterized queries are essential:**

With the vulnerable version, an attacker could input:
```
user_id = "1; DROP TABLE users; --"
```

This would execute: `SELECT * FROM users WHERE id = 1; DROP TABLE users; --`

Parameterized queries:
- Separate SQL logic from data
- Database driver handles escaping
- Impossible to inject SQL commands
- Also more efficient (query plan caching)""",
                "category": "sql_injection",
                "severity": "critical",
            },

            # Input validation
            {
                "prompt": "How do I handle user input in a web API?",
                "rejected": """Just use the input directly:

```python
@app.route('/api/user/<user_id>')
def get_user(user_id):
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")
```""",
                "chosen": """Validate and sanitize all user input:

```python
from pydantic import BaseModel, validator, constr
from typing import Optional

class UserQuery(BaseModel):
    user_id: int  # Validates it's an integer
    fields: Optional[list[str]] = None

    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('user_id must be positive')
        return v

    @validator('fields')
    def validate_fields(cls, v):
        allowed = {'id', 'name', 'email', 'created_at'}
        if v and not set(v).issubset(allowed):
            raise ValueError(f'fields must be subset of {allowed}')
        return v

@app.route('/api/user/<int:user_id>')
def get_user(user_id: int):
    # Flask's int: converter validates the type
    query = UserQuery(user_id=user_id)
    return db.get_user(query.user_id)
```

**Defense in depth:**
- Type conversion at route level (`<int:user_id>`)
- Pydantic validation for business rules
- Parameterized queries at database level
- Never trust user input, even from authenticated users""",
                "category": "input_validation",
                "severity": "high",
            },
        ]

        # ============ DOCKER ANTI-PATTERNS ============
        self.docker_patterns = [
            {
                "prompt": "How do I write a Dockerfile for a Python app?",
                "rejected": """Here's a simple Dockerfile:

```dockerfile
FROM python:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py
```""",
                "chosen": """Use a multi-stage build with best practices:

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy only the wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Use exec form for proper signal handling
CMD ["python", "app.py"]
```

**Why this is better:**
- **Pinned version**: `python:3.11-slim` not `latest` (reproducible builds)
- **Multi-stage**: Smaller final image (no build tools)
- **Non-root user**: Security best practice
- **No cache**: Smaller image size
- **Exec form CMD**: Proper signal handling (graceful shutdown)
- **Layer optimization**: Dependencies before code (better caching)""",
                "category": "docker_best_practices",
                "severity": "high",
            },
        ]

    def generate_all_pairs(self) -> List[DPOPair]:
        """Generate all DPO pairs"""
        pairs = []

        all_patterns = (
            self.python_patterns +
            self.kubernetes_patterns +
            self.security_patterns +
            self.docker_patterns
        )

        for pattern in all_patterns:
            pairs.append(DPOPair(
                prompt=pattern["prompt"],
                chosen=pattern["chosen"],
                rejected=pattern["rejected"],
                category=pattern["category"],
                severity=pattern.get("severity", "medium"),
            ))

        return pairs

    def to_dpo_format(self, pairs: List[DPOPair]) -> List[Dict]:
        """Convert to standard DPO training format"""
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in pairs
        ]

    def to_orpo_format(self, pairs: List[DPOPair]) -> List[Dict]:
        """Convert to ORPO (Odds Ratio Preference Optimization) format"""
        return [
            {
                "instruction": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in pairs
        ]

    def to_rlhf_format(self, pairs: List[DPOPair]) -> List[Dict]:
        """Convert to RLHF comparison format"""
        data = []
        for p in pairs:
            data.append({
                "prompt": p.prompt,
                "responses": [p.chosen, p.rejected],
                "ranking": [0, 1],  # 0 is preferred
            })
        return data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO training data")
    parser.add_argument("--output", default="data/training/dpo_pairs.json")
    parser.add_argument("--format", choices=["dpo", "orpo", "rlhf"], default="dpo")

    args = parser.parse_args()

    generator = DPODataGenerator()
    pairs = generator.generate_all_pairs()

    if args.format == "dpo":
        data = generator.to_dpo_format(pairs)
    elif args.format == "orpo":
        data = generator.to_orpo_format(pairs)
    else:
        data = generator.to_rlhf_format(pairs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(pairs)} DPO pairs")
    print(f"Saved to {output_path}")

    # Stats by category
    categories = {}
    for p in pairs:
        categories[p.category] = categories.get(p.category, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
