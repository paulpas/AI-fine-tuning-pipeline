# Training Results: python-expert-v5

**Generated:** 2026-01-20 11:43:28

## Summary

| Metric | Fine-tuned | Base Model | Improvement |
|--------|------------|------------|-------------|
| **Average Score** | 88.7/100 | 81.7/100 | +6.9 (+8.5%) |
| **Avg Response Time** | 22.8s | 24.1s | -1.3s |
| **Tests with Code** | 15/15 | 12/15 | +3 |
| **Tests Improved** | 7/15 | - | - |
| **Tests Regressed** | 2/15 | - | - |

## Models Compared

| | Model |
|--|-------|
| **Fine-tuned** | `python-expert-v5` |
| **Base** | `deepseek-r1:1.5b` |

## Training Data Sources

| Source | Count |
|--------|-------|
| Stack Overflow Tags | 11 |
| GitHub Static Repos | 13 |
| GitHub Search Queries | 111 |

## Results by Category

| Category | Fine-tuned Avg | Base Model Avg | Improvement |
|----------|----------------|----------------|-------------|
| AWS | 84.0 | 76.0 | ✅ +8.0 |
| AWS Security | 92.0 | 54.0 | ✅ +38.0 |
| CI/CD | 90.0 | 100.0 | ❌ -10.0 |
| Crossplane | 84.0 | 62.0 | ✅ +22.0 |
| Docker | 92.0 | 92.0 | ➖ +0.0 |
| GitOps | 76.0 | 76.0 | ➖ +0.0 |
| Kubernetes | 96.0 | 92.0 | ➖ +4.0 |
| Network Security | 68.0 | 68.0 | ➖ +0.0 |
| Observability | 92.0 | 62.0 | ✅ +30.0 |
| Platform Engineering | 84.0 | 84.0 | ➖ +0.0 |
| Python | 100.0 | 100.0 | ➖ +0.0 |
| Python Async | 84.0 | 76.0 | ✅ +8.0 |
| Terraform/IaC | 100.0 | 92.0 | ✅ +8.0 |
| Testing | 92.0 | 100.0 | ❌ -8.0 |

---

## Before/After Examples

### Most Improved Examples

#### AWS Security: How do I implement AWS KMS key policies with least privilege...

**Improvement:** 54 → 92 (+38 points)

<details>
<summary>📗 Fine-tuned Response (Score: 92)</summary>

```
<think>
Okay, so the user wants to implement AWS KMS key policies with minimal privileges access. Let me break down their request step by step. They're probably in a development environment and need secure access controls for their application's keys. 

First, I should explain what KMS policies are and why they matter. Using them can prevent unauthorized access to sensitive data like API endpoints or private keys. But since they want minimal privilege, the user might not fully understand how to set up these policies.

I'll start by addressing their initial question: how to create a basic policy with least privilege access using Cognito API. This will help them get started without getting overwhelmed by complex settings. I'll outline the steps clearly and provide code snippets so they can follow along easily.

Next, it's important to explain why KMS policies are valuable in this context. By ensuring that only authorized users have access, their system becomes more secure. They might not realize how critical data security is in modern applications, especially with sensitive information like API endpoints or decryption keys.

I should also guide them on how to test the policy to make sure it works as intended. This hands-on example will help them see the benefits of using policies immediately after setting up their environment.

Additionally, I'll highlight why they shouldn't manually set permissions. It's better for consistency and security if users already understand that by d...[truncated]
```

</details>

<details>
<summary>📕 Base Model Response (Score: 54)</summary>

```
Thinking...
Okay, so I need to figure out how to implement AWS KMS key policies with least privilege access. I'm not very familiar with this topic, but I'll try to break it down step by step.

First, I know that KMS is Amazon's cloud security service used for encryption keys and certificates. It helps protect sensitive data at rest and in transit. Key policies in KMS allow you to define which users or roles have access to certain keys.

The user mentioned least privilege access, so I guess that means making sure the key policies are set up in a way that only specific people can use these keys. But how do I determine what constitutes "least privilege"? It's subjective, but generally, it's about controlling who can see or use the keys without unnecessary exposure.

I think KMS allows specifying access using groups and roles. So maybe I should start by defining which groups have access to certain keys. Each group is like a set of people who are together, so the key policies will be shared among them. But how do I get started with this?

Let me recall that in AWS, you can create groups, roles, and users. The user role probably includes all accounts on KMS. Then, there might be group roles or multiuser groups where only specific groups can access keys. And maybe even individual account roles for more granular control.

I should check if I need to set up something like multiuser groups. From what I remember, a multiuser group is a subgroup of the user role that includes a subset of...[truncated]
```

</details>

---

#### Observability: How do I set up Prometheus and Grafana on Kubernetes using t...

**Improvement:** 62 → 92 (+30 points)

<details>
<summary>📗 Fine-tuned Response (Score: 92)</summary>

```
<think>
Alright, so the user is asking about setting up Prometheus and Grafana on Kubernetes using thekube-prometheus-stack Helm chart. They mentioned they have a setup with Prometheus, Grafana, and Kubernetes running separately.

First, I need to understand what Helm chart is in this context. From what I know, Helm chart is a tool that helps manage and monitor Kubernetes clusters by providing an easy-to-use CLI for building and monitoring the cluster.

The user probably wants to combine Prometheus and Grafana into the Kubernetes container. They might be looking to have a centralized dashboard and monitoring system within their Kubernetes app using cloud services like Prometheus and Grafana, all on top of Helm chart.

I should explain how to install Helm chart along with Prometheus and Grafana. Then, describe each step clearly: installing dependencies, setting up Kubernetes cluster, adding Helm chart commands, configuring Prometheus, installing Grafana CLI tools in the Helm chart container, running it, and configuring the dashboard settings.

It's important to mention the benefits of this setup, such as centralized monitoring, ease of use, and scalability. I should also provide instructions on how to access the dashboard from Kubernetes services using Prometheus commands.

Finally, I'll offer help if they have any specific issues or questions about their setup.
</think>

Great! You're looking at a common Kubernetes setup that combines **Prometheus** and **Grafana** with Helm ...[truncated]
```

</details>

<details>
<summary>📕 Base Model Response (Score: 62)</summary>

```
Thinking...
Okay, so I need to figure out how to set up Prometheus and Grafana on Kubernetes usingkubePrometheusStack.Helm. I'm new to this, so I'll have to take it step by step.

First, I think I should install Python on my system because the Helm command-line tools use Python. I remember hearing that Python 3 is recommended. So I'll download Python from the official site and run the setup.py script. Once that's done, installing pip might be necessary for some of the libraries, so I'll need to do that as well.

After installation, I should look at the example files available in the Helm directory. The example.kubefile looks like a good starting point. It uses Kubernetes, Prometheus, Grafana, and other dependencies, which are all what I need. I'll copy that file into my work folder, right?

Next, running kube promote --version will display the version information. But since I'm on Windows, I think it might just show an empty message if there's a problem. Maybe I can check the output to see if anything goes wrong.

Now, for setting up Prometheus. I know that Prometheus is part of the Helm tools, so I'll need to start the Prometheus worker and the cluster. The commands seem straightforward: running "prometheus worker" starts one worker, and then "prometheus cluster" to connect it to the prometheus worker. 

I'm not sure how the worker connects to the prometheus worker or if there are any configurations I should make for Prometheus settings like interval or data sources. Maybe I...[truncated]
```

</details>

---

#### Crossplane: How do I use Crossplane compositions to provision AWS RDS in...

**Improvement:** 62 → 84 (+22 points)

<details>
<summary>📗 Fine-tuned Response (Score: 84)</summary>

```
<think>
Okay, so the user asked about using CrossPlane for provision of AWS RDS instances. I know that CrossPlane is a popular CLI tool for setting up serverless applications in AWS with Python. They might be new to this and want a clear guide.

First, I should start by explaining what AWS RDS is because without knowing it, the answer won't make much sense. So, I'll briefly mention Amazon's Relational Database Service which allows users to store data in PostgreSQL or MySQL in an EC2 managed environment.

Next, I need to outline the steps to set up a single instance using CrossPlane. It would be helpful to include examples of relevant commands like `crossplane setup`, `create`, and `add_to`. Maybe also mention how to check if it's running with `check_status` so the user knows they can continue.

Then, I should explain how to provision multiple instances. This is useful because servers are often needed for a web app. Including examples of creating several instances using commands like `create` followed by numbers would be helpful.

I also want to cover what happens if they need more instances after starting one, so it's good to mention that you can use the same command with additional arguments.

Finally, I should talk about troubleshooting common issues like errors or permissions not being granted. This shows the user why they might face problems and how to troubleshoot them.

Overall, the response needs to be clear, concise, and cover all necessary steps without overwhelming ...[truncated]
```

</details>

<details>
<summary>📕 Base Model Response (Score: 62)</summary>

```
Thinking...
Okay, so I want to figure out how to use CrossPlane Compositions to provision AWS RDS instances. Hmm, I'm not entirely sure where to start, but I'll try to break it down step by step.

First, I know that AWS RDS stands for Amazon Redshift and is used for big data analytics. So, the goal here is probably to create a service that can store large amounts of data. Now, CrossPlane Compositions are a tool for building applications in an IoT environment, so maybe they'll help in managing the database.

I remember that using CrossPlane, you can build services like RDS by creating containers or functions. But wait, how exactly do I set up RDS within a container? Maybe I need to use Docker somehow because containers are a common way to deploy applications in AWS?

So, if I'm setting up an RDS instance inside a containerized service (Container Service), the steps would probably involve defining the container as part of the Service. But I'm not entirely clear on how to structure that in CrossPlane. Maybe I need to set up a container that runs Docker with a specific build image?

Wait, I think I saw something about using Docker Compose for this. So maybe I should create a service that uses a Docker build script. That way, the container can be deployed and managed within a containerized environment.

Next, how do I actually create an RDS instance in a container? I know you can't run a database server in a Docker image directly because it's not compatible with AWS's CI/CD pipeli...[truncated]
```

</details>

---

## Detailed Test Results

| # | Category | Prompt | Fine-tuned | Base | Diff | Status |
|---|----------|--------|------------|------|------|--------|
| 1 | Kubernetes | How do I create a Kubernetes deployment ... | 100 | 92 | +8 | ➖ Similar |
| 2 | Kubernetes | Write Python code to watch for pod event... | 92 | 92 | +0 | ➖ Similar |
| 3 | Docker | How do I build and push a Docker image u... | 92 | 92 | +0 | ➖ Similar |
| 4 | Terraform/IaC | Show me a Terraform module for creating ... | 100 | 92 | +8 | ➖ Similar |
| 5 | Crossplane | How do I use Crossplane compositions to ... | 84 | 62 | +22 | ✅ Improved |
| 6 | AWS | Write Python boto3 code to assume a role... | 84 | 76 | +8 | ➖ Similar |
| 7 | AWS Security | How do I implement AWS KMS key policies ... | 92 | 54 | +38 | ✅ Improved |
| 8 | Python | Show me how to properly handle exception... | 100 | 100 | +0 | ➖ Similar |
| 9 | Python Async | Write a Python async function that fetch... | 84 | 76 | +8 | ➖ Similar |
| 10 | Testing | How do I write pytest fixtures for testi... | 92 | 100 | -8 | ➖ Similar |
| 11 | CI/CD | Create a GitHub Actions workflow that de... | 90 | 100 | -10 | ➖ Similar |
| 12 | GitOps | Show me an ArgoCD ApplicationSet that ge... | 76 | 76 | +0 | ➖ Similar |
| 13 | Network Security | Write a Kubernetes NetworkPolicy that al... | 68 | 68 | +0 | ➖ Similar |
| 14 | Observability | How do I set up Prometheus and Grafana o... | 92 | 62 | +30 | ✅ Improved |
| 15 | Platform Engineering | How do I create a Backstage software tem... | 84 | 84 | +0 | ➖ Similar |

---

## Conclusion

✅ **MODERATE IMPROVEMENT** - The fine-tuned model shows meaningful gains.

### Recommendations

Consider additional training data to further improve performance.

### Key Statistics

- **Total Tests:** 15
- **Average Improvement:** +22.0 points (+8.5%)
- **Best Category:** Terraform/IaC
- **Most Improved:** AWS Security

---

*Generated by `scripts/evaluate_model.py` on 2026-01-20 11:43:28*
