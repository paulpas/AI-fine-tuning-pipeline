#!/usr/bin/env python3
"""
DevOps Training Data Collector

Collects high-quality training data from multiple sources:
1. Stack Overflow Q&A (Python, Kubernetes, Docker, Terraform)
2. GitHub Issues and Discussions
3. Official documentation
4. DevOps blogs and tutorials

This creates diverse, high-quality instruction/response pairs that teach
practical DevOps skills.
"""

import json
import os
import re
import time
import hashlib
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Generator
from datetime import datetime, timedelta
import html


@dataclass
class QAPair:
    """A question-answer training pair"""
    instruction: str
    input: str
    output: str
    source: str
    score: int = 0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_alpaca(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }


class StackOverflowCollector:
    """
    Collects high-quality Q&A from Stack Overflow

    Focuses on:
    - Highly upvoted answers
    - Python + DevOps related tags
    - Clear code examples
    """

    BASE_URL = "https://api.stackexchange.com/2.3"
    DEVOPS_TAGS = [
        "python",
        "kubernetes",
        "docker",
        "terraform",
        "aws",
        "gcp",
        "azure",
        "ansible",
        "jenkins",
        "github-actions",
        "gitlab-ci",
        "helm",
        "prometheus",
        "grafana",
        "flask",
        "fastapi",
        "django",
        "pytest",
        "asyncio",
        "boto3",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("STACKOVERFLOW_API_KEY")
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        params["site"] = "stackoverflow"
        params["filter"] = "withbody"  # Include body content

        if self.api_key:
            params["key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)

        # Rate limiting
        if response.status_code == 429:
            backoff = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting {backoff} seconds...")
            time.sleep(backoff)
            return self._make_request(endpoint, params)

        response.raise_for_status()
        return response.json()

    def get_top_questions(
        self,
        tag: str,
        min_score: int = 10,
        page_size: int = 100,
        max_pages: int = 5
    ) -> Generator[Dict, None, None]:
        """Get top-scored questions for a tag"""

        for page in range(1, max_pages + 1):
            params = {
                "tagged": tag,
                "sort": "votes",
                "order": "desc",
                "pagesize": page_size,
                "page": page,
            }

            try:
                data = self._make_request("questions", params)
            except Exception as e:
                print(f"Error fetching questions for {tag}: {e}")
                break

            for question in data.get("items", []):
                if question.get("score", 0) >= min_score:
                    yield question

            if not data.get("has_more", False):
                break

            time.sleep(0.5)  # Be nice to the API

    def get_accepted_answer(self, question_id: int) -> Optional[Dict]:
        """Get the accepted answer for a question"""
        params = {
            "order": "desc",
            "sort": "votes",
        }

        try:
            data = self._make_request(f"questions/{question_id}/answers", params)
            answers = data.get("items", [])

            # Prefer accepted answer, fall back to highest voted
            for answer in answers:
                if answer.get("is_accepted", False):
                    return answer

            return answers[0] if answers else None
        except Exception as e:
            print(f"Error fetching answer for question {question_id}: {e}")
            return None

    def clean_html(self, text: str) -> str:
        """Convert HTML to clean markdown-like text"""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Convert code blocks
        text = re.sub(r'<pre><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', text, flags=re.DOTALL)
        text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)

        # Convert lists
        text = re.sub(r'<li>(.*?)</li>', r'- \1\n', text)
        text = re.sub(r'<[ou]l>', '', text)
        text = re.sub(r'</[ou]l>', '', text)

        # Convert paragraphs and breaks
        text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
        text = re.sub(r'<br\s*/?>', '\n', text)

        # Convert links
        text = re.sub(r'<a href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text)

        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def extract_qa_pairs(self, tag: str, limit: int = 500) -> List[QAPair]:
        """Extract Q&A pairs for a specific tag"""
        pairs = []

        print(f"Collecting Stack Overflow Q&A for tag: {tag}")

        for i, question in enumerate(self.get_top_questions(tag)):
            if len(pairs) >= limit:
                break

            answer = self.get_accepted_answer(question["question_id"])
            if not answer:
                continue

            # Clean the content
            q_body = self.clean_html(question.get("body", ""))
            a_body = self.clean_html(answer.get("body", ""))

            # Skip if answer is too short
            if len(a_body) < 100:
                continue

            # Create instruction from title
            title = question.get("title", "")
            instruction = self._title_to_instruction(title)

            pair = QAPair(
                instruction=instruction,
                input=q_body if len(q_body) > 50 else "",
                output=a_body,
                source=f"stackoverflow:{question['question_id']}",
                score=answer.get("score", 0),
                tags=question.get("tags", []),
            )
            pairs.append(pair)

            if (i + 1) % 50 == 0:
                print(f"  Collected {len(pairs)} pairs...")

            time.sleep(0.2)

        return pairs

    def _title_to_instruction(self, title: str) -> str:
        """Convert question title to natural instruction"""
        # Clean up the title
        title = title.strip()

        # If already a question, use as-is
        if title.endswith("?"):
            return title

        # Convert to question form
        question_starters = [
            "How do I ", "How to ", "What is ", "Why does ",
            "Can I ", "Should I ", "Is it possible to "
        ]

        for starter in question_starters:
            if title.lower().startswith(starter.lower()):
                return title + "?"

        # Default: prepend "How do I"
        return f"How do I {title.lower()}?"


class GitHubCollector:
    """
    Collects training data from GitHub:
    - Issues with good solutions
    - Discussions
    - Code examples from READMEs
    """

    BASE_URL = "https://api.github.com"
    DEVOPS_REPOS = [
        "kubernetes/kubernetes",
        "docker/docker-py",
        "hashicorp/terraform",
        "ansible/ansible",
        "pytest-dev/pytest",
        "pallets/flask",
        "tiangolo/fastapi",
        "boto/boto3",
        "kubernetes-client/python",
        "prometheus/client_python",
    ]

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"

    def get_issues_with_solutions(
        self,
        repo: str,
        label: str = "good first issue",
        limit: int = 100
    ) -> List[QAPair]:
        """Get closed issues that have helpful comments"""
        pairs = []

        params = {
            "state": "closed",
            "sort": "comments",
            "direction": "desc",
            "per_page": min(limit, 100),
        }

        url = f"{self.BASE_URL}/repos/{repo}/issues"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            issues = response.json()
        except Exception as e:
            print(f"Error fetching issues from {repo}: {e}")
            return pairs

        for issue in issues:
            if issue.get("pull_request"):
                continue  # Skip PRs

            # Get comments
            comments_url = issue.get("comments_url")
            if not comments_url:
                continue

            try:
                time.sleep(0.5)
                response = self.session.get(comments_url)
                comments = response.json()
            except:
                continue

            # Find the most upvoted comment
            best_comment = None
            for comment in comments:
                reactions = comment.get("reactions", {})
                score = reactions.get("+1", 0) + reactions.get("heart", 0)
                if not best_comment or score > best_comment.get("_score", 0):
                    comment["_score"] = score
                    best_comment = comment

            if best_comment and len(best_comment.get("body", "")) > 100:
                pair = QAPair(
                    instruction=issue.get("title", ""),
                    input=issue.get("body", "")[:500] if issue.get("body") else "",
                    output=best_comment.get("body", ""),
                    source=f"github:{repo}:{issue['number']}",
                    score=best_comment.get("_score", 0),
                    tags=[repo.split("/")[0]],
                )
                pairs.append(pair)

            if len(pairs) >= limit:
                break

        return pairs


class DocumentationExtractor:
    """
    Extracts training data from official documentation
    """

    DOC_REPOS = {
        "kubernetes": {
            "url": "https://github.com/kubernetes/website",
            "docs_path": "content/en/docs",
            "pattern": r"```python\n(.*?)```",
        },
        "terraform": {
            "url": "https://github.com/hashicorp/terraform-docs-common",
            "docs_path": "website/docs",
            "pattern": r"```hcl\n(.*?)```",
        },
    }

    def extract_code_examples(self, markdown_content: str) -> List[Dict]:
        """Extract code examples from markdown"""
        examples = []

        # Find Python code blocks
        python_pattern = r'```python\n(.*?)```'
        matches = re.findall(python_pattern, markdown_content, re.DOTALL)

        for code in matches:
            if len(code) > 50:  # Skip trivial examples
                examples.append({
                    "code": code.strip(),
                    "language": "python"
                })

        return examples


class DevOpsQAGenerator:
    """
    Generates Q&A pairs from templates for common DevOps tasks
    """

    TEMPLATES = {
        "kubernetes": [
            {
                "instruction": "How do I create a Kubernetes deployment programmatically in Python?",
                "output": """Here's how to create a Kubernetes deployment using the Python client:

```python
from kubernetes import client, config

def create_deployment(name: str, image: str, replicas: int = 1, namespace: str = "default"):
    \"\"\"Create a Kubernetes deployment.\"\"\"
    config.load_kube_config()

    # Define container
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=80)]
    )

    # Define pod template
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=client.V1PodSpec(containers=[container])
    )

    # Define deployment spec
    spec = client.V1DeploymentSpec(
        replicas=replicas,
        selector=client.V1LabelSelector(match_labels={"app": name}),
        template=template
    )

    # Create deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec
    )

    # Create in cluster
    apps_v1 = client.AppsV1Api()
    return apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)

# Usage
deployment = create_deployment("nginx", "nginx:1.21", replicas=3)
print(f"Deployment {deployment.metadata.name} created")
```

**Key points:**
- Always use `config.load_kube_config()` for local development or `config.load_incluster_config()` inside a pod
- Use typed objects (`V1Container`, `V1Deployment`) for better IDE support
- The label selector must match the pod template labels""",
            },
            {
                "instruction": "How do I watch Kubernetes events in real-time with Python?",
                "output": """Use the Watch API for efficient real-time monitoring:

```python
from kubernetes import client, config, watch

def watch_pods(namespace: str = "default", timeout: int = 300):
    \"\"\"Watch pod events in real-time.\"\"\"
    config.load_kube_config()
    v1 = client.CoreV1Api()
    w = watch.Watch()

    try:
        for event in w.stream(
            v1.list_namespaced_pod,
            namespace=namespace,
            timeout_seconds=timeout
        ):
            pod = event['object']
            event_type = event['type']

            print(f"{event_type}: {pod.metadata.name} - {pod.status.phase}")

            # Handle specific events
            if event_type == "DELETED":
                print(f"  Pod {pod.metadata.name} was deleted")
            elif pod.status.phase == "Failed":
                print(f"  Pod {pod.metadata.name} failed!")

    except Exception as e:
        print(f"Watch error: {e}")
    finally:
        w.stop()

# Usage
watch_pods(namespace="default", timeout=60)
```

**Why use Watch instead of polling:**
- More efficient (no repeated API calls)
- Real-time updates (no delay)
- Lower API server load
- Handles reconnection automatically""",
            },
        ],
        "docker": [
            {
                "instruction": "How do I build and push a Docker image with Python?",
                "output": """Use the docker-py library to programmatically build and push images:

```python
import docker
from docker.errors import BuildError, APIError

def build_and_push(
    dockerfile_path: str,
    image_name: str,
    tag: str = "latest",
    registry: str = None
):
    \"\"\"Build a Docker image and push to registry.\"\"\"
    client = docker.from_env()

    full_tag = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"

    print(f"Building {full_tag}...")

    try:
        # Build image
        image, build_logs = client.images.build(
            path=dockerfile_path,
            tag=full_tag,
            rm=True,  # Remove intermediate containers
            pull=True,  # Pull base image updates
        )

        # Print build logs
        for chunk in build_logs:
            if 'stream' in chunk:
                print(chunk['stream'], end='')

        print(f"\\nSuccessfully built {image.id[:12]}")

        # Push to registry
        if registry:
            print(f"Pushing to {registry}...")
            for line in client.images.push(full_tag, stream=True, decode=True):
                if 'status' in line:
                    print(line['status'])
            print("Push complete!")

        return image

    except BuildError as e:
        print(f"Build failed: {e}")
        for log in e.build_log:
            if 'stream' in log:
                print(log['stream'], end='')
        raise
    except APIError as e:
        print(f"Docker API error: {e}")
        raise

# Usage
build_and_push("./app", "myapp", tag="v1.0.0", registry="docker.io/myuser")
```

**Best practices:**
- Always use specific tags, not `latest`
- Use `rm=True` to clean up intermediate layers
- Handle errors gracefully with specific exception types""",
            },
        ],
        "terraform": [
            {
                "instruction": "How do I run Terraform commands from Python?",
                "output": """Use python-terraform or subprocess for Terraform automation:

```python
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional

class TerraformRunner:
    \"\"\"Run Terraform commands programmatically.\"\"\"

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)

    def _run(self, *args, capture_json: bool = False) -> Dict:
        \"\"\"Run a terraform command.\"\"\"
        cmd = ["terraform", *args]

        if capture_json:
            cmd.append("-json")

        result = subprocess.run(
            cmd,
            cwd=self.working_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Terraform failed: {result.stderr}")

        if capture_json:
            return json.loads(result.stdout)
        return {"stdout": result.stdout, "stderr": result.stderr}

    def init(self, backend_config: Optional[Dict] = None) -> Dict:
        \"\"\"Initialize Terraform.\"\"\"
        args = ["init", "-input=false"]

        if backend_config:
            for key, value in backend_config.items():
                args.append(f"-backend-config={key}={value}")

        return self._run(*args)

    def plan(self, var_file: Optional[str] = None, out: str = "tfplan") -> Dict:
        \"\"\"Create execution plan.\"\"\"
        args = ["plan", f"-out={out}", "-input=false"]

        if var_file:
            args.append(f"-var-file={var_file}")

        return self._run(*args)

    def apply(self, plan_file: str = "tfplan", auto_approve: bool = False) -> Dict:
        \"\"\"Apply the plan.\"\"\"
        args = ["apply", "-input=false"]

        if auto_approve:
            args.append("-auto-approve")
        else:
            args.append(plan_file)

        return self._run(*args)

    def output(self) -> Dict:
        \"\"\"Get Terraform outputs.\"\"\"
        return self._run("output", capture_json=True)

    def destroy(self, auto_approve: bool = False) -> Dict:
        \"\"\"Destroy infrastructure.\"\"\"
        args = ["destroy", "-input=false"]
        if auto_approve:
            args.append("-auto-approve")
        return self._run(*args)

# Usage
tf = TerraformRunner("./infrastructure")
tf.init()
tf.plan(var_file="prod.tfvars")
tf.apply()
outputs = tf.output()
print(f"Instance IP: {outputs['instance_ip']['value']}")
```

**Key considerations:**
- Always use `-input=false` for automation
- Use plan files for safe applies
- Capture and validate outputs
- Consider using workspaces for environments""",
            },
        ],
        "debugging": [
            {
                "instruction": "I'm getting 'kubernetes.client.exceptions.ApiException: (403) Forbidden' - how do I fix this?",
                "output": """This error means your service account lacks permission for the requested operation.

**Diagnosis:**
```python
from kubernetes import client, config
from kubernetes.client.rest import ApiException

config.load_kube_config()
v1 = client.CoreV1Api()

try:
    pods = v1.list_namespaced_pod("default")
except ApiException as e:
    print(f"Status: {e.status}")
    print(f"Reason: {e.reason}")
    print(f"Body: {e.body}")
```

**Common causes and fixes:**

1. **Missing RBAC permissions** - Create a ClusterRole/RoleBinding:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: pod-reader-binding
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
roleRef:
  kind: ClusterRole
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

2. **Wrong context** - Check your kubeconfig:
```python
contexts, active_context = config.list_kube_config_contexts()
print(f"Active context: {active_context['name']}")
```

3. **Expired token** - Re-authenticate:
```bash
# For GKE
gcloud container clusters get-credentials CLUSTER_NAME

# For EKS
aws eks update-kubeconfig --name CLUSTER_NAME
```

**Prevention:**
- Use namespace-scoped roles when possible (least privilege)
- Test permissions with `kubectl auth can-i list pods`
- Log the specific operation that fails for debugging""",
            },
        ],
    }

    def get_all_templates(self) -> List[QAPair]:
        """Get all template Q&A pairs"""
        pairs = []

        for domain, templates in self.TEMPLATES.items():
            for template in templates:
                pair = QAPair(
                    instruction=template["instruction"],
                    input="",
                    output=template["output"],
                    source=f"template:{domain}",
                    score=100,  # High quality templates
                    tags=[domain],
                )
                pairs.append(pair)

        return pairs


def collect_all_data(output_dir: str, so_api_key: str = None, gh_token: str = None):
    """Collect data from all sources"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # 1. Collect from Stack Overflow
    if so_api_key or os.environ.get("STACKOVERFLOW_API_KEY"):
        print("\n=== Collecting from Stack Overflow ===")
        so_collector = StackOverflowCollector(so_api_key)

        for tag in ["python-kubernetes", "docker-py", "boto3", "terraform", "ansible"]:
            try:
                pairs = so_collector.extract_qa_pairs(tag, limit=200)
                print(f"  {tag}: {len(pairs)} pairs")
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"  Error with {tag}: {e}")

    # 2. Collect from GitHub
    if gh_token or os.environ.get("GITHUB_TOKEN"):
        print("\n=== Collecting from GitHub ===")
        gh_collector = GitHubCollector(gh_token)

        for repo in GitHubCollector.DEVOPS_REPOS[:5]:  # Limit for demo
            try:
                pairs = gh_collector.get_issues_with_solutions(repo, limit=50)
                print(f"  {repo}: {len(pairs)} pairs")
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"  Error with {repo}: {e}")

    # 3. Add curated templates
    print("\n=== Adding curated templates ===")
    template_gen = DevOpsQAGenerator()
    template_pairs = template_gen.get_all_templates()
    print(f"  Templates: {len(template_pairs)} pairs")
    all_pairs.extend(template_pairs)

    # Deduplicate
    seen = set()
    deduped = []
    for pair in all_pairs:
        key = hashlib.md5((pair.instruction + pair.output).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(pair)

    print(f"\n=== Total: {len(deduped)} unique pairs ===")

    # Save in multiple formats
    alpaca_data = [p.to_alpaca() for p in deduped]

    with open(output_path / "devops_training_data.json", "w") as f:
        json.dump(alpaca_data, f, indent=2)

    # Save with metadata for filtering
    full_data = [asdict(p) for p in deduped]
    with open(output_path / "devops_training_data_full.json", "w") as f:
        json.dump(full_data, f, indent=2)

    print(f"\nSaved to {output_path}")
    return deduped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect DevOps training data")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--so-key", help="Stack Overflow API key")
    parser.add_argument("--gh-token", help="GitHub token")

    args = parser.parse_args()

    collect_all_data(args.output, args.so_key, args.gh_token)
