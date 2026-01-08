"""Example demonstrating automated canary validation in CI/CD.

This script shows how to integrate canary validation into your
GitHub Actions deployment pipeline.
"""
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from loguru import logger

from src.app.deployment.canary_analyzer import CanaryAnalyzer, CanaryMetrics
from src.app.deployment.rollback_policy import (
    RollbackPolicy,
    RollbackDecision,
    RollbackThresholds,
    evaluate_canary
)


class HelmManager:
    """Manages Helm deployments for canary validation."""
    
    def __init__(self, release_name: str = "apex-inference"):
        """Initialize Helm manager.
        
        Args:
            release_name: Helm release name
        """
        self.release_name = release_name
        self.canary_release = f"{release_name}-canary"
    
    async def deploy_canary(self, version: str, values_file: str) -> bool:
        """Deploy canary version using Helm.
        
        Args:
            version: Version tag to deploy
            values_file: Path to Helm values file
            
        Returns:
            True if deployment successful
        """
        try:
            cmd = [
                "helm", "upgrade", "--install", self.canary_release,
                "./charts/apex-inference",
                "--version", version,
                "--values", values_file,
                "--set", "image.tag=" + version,
                "--set", "canary.enabled=true",
                "--wait", "--timeout=10m"
            ]
            
            logger.info(f"Deploying canary version {version}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Helm deployment failed: {result.stderr}")
                return False
            
            logger.info("Canary deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy canary: {e}")
            return False
    
    async def rollback(self, revision: Optional[int] = None) -> bool:
        """Rollback to previous revision.
        
        Args:
            revision: Specific revision to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            cmd = ["helm", "rollback", self.release_name]
            if revision:
                cmd.append(str(revision))
            
            logger.info(f"Rolling back {self.release_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Helm rollback failed: {result.stderr}")
                return False
            
            logger.info("Rollback successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False
    
    async def promote_canary(self) -> bool:
        """Promote canary to production.
        
        Returns:
            True if promotion successful
        """
        try:
            # Update production to canary version
            cmd = [
                "helm", "upgrade", self.release_name,
                "./charts/apex-inference",
                "--reuse-values",
                "--set", "canary.enabled=false"
            ]
            
            logger.info("Promoting canary to production...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Helm promotion failed: {result.stderr}")
                return False
            
            # Delete canary release
            delete_cmd = ["helm", "delete", self.canary_release]
            subprocess.run(delete_cmd, capture_output=True, text=True)
            
            logger.info("Canary promotion successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            return False
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status.
        
        Returns:
            Dictionary with deployment information
        """
        try:
            # Get helm status
            cmd = ["helm", "status", self.release_name, "--output", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"status": "not_found"}
            
            status = json.loads(result.stdout)
            
            return {
                "status": "deployed",
                "version": status.get("version", "unknown"),
                "revision": status.get("version", "unknown"),
                "updated_at": status.get("updated", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"status": "error", "error": str(e)}


class AlertManager:
    """Manages alerts for canary validation."""
    
    def __init__(self, slack_webhook_url: Optional[str] = None):
        """Initialize alert manager.
        
        Args:
            slack_webhook_url: Slack webhook URL for notifications
        """
        self.slack_webhook_url = slack_webhook_url
    
    async def send_alert(self, message: str, color: str = "danger") -> bool:
        """Send alert to Slack.
        
        Args:
            message: Alert message
            color: Message color (danger, warning, good)
            
        Returns:
            True if alert sent successfully
        """
        if not self.slack_webhook_url:
            logger.warning("No Slack webhook configured")
            return False
        
        try:
            payload = {
                "attachments": [{
                    "color": color,
                    "text": message,
                    "ts": int(datetime.now(timezone.utc).timestamp())
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.slack_webhook_url,
                    json=payload
                )
                response.raise_for_status()
            
            logger.info("Alert sent to Slack")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    async def notify_success(self, version: str) -> None:
        """Notify about successful deployment.
        
        Args:
            version: Deployed version
        """
        message = f"✅ Canary validation passed for version {version}"
        await self.send_alert(message, color="good")
    
    async def notify_rollback(self, version: str, reason: str) -> None:
        """Notify about automatic rollback.
        
        Args:
            version: Version being rolled back
            reason: Reason for rollback
        """
        message = f"🚨 Automatic rollback triggered for {version}\n\nReason: {reason}"
        await self.send_alert(message, color="danger")
    
    async def notify_manual_review(self, version: str, reasoning: Dict[str, Any]) -> None:
        """Notify about manual review requirement.
        
        Args:
            version: Version requiring review
            reasoning: Decision reasoning
        """
        message = f"⚠️ Manual review required for {version}\n\n"
        message += f"Primary reason: {reasoning.get('primary_reason', 'Unknown')}\n"
        
        if reasoning.get("triggered_conditions"):
            message += "\nConditions:\n"
            for condition in reasoning["triggered_conditions"]:
                message += f"- {condition}\n"
        
        await self.send_alert(message, color="warning")


class CanaryValidator:
    """Orchestrates the complete canary validation process."""
    
    def __init__(
        self,
        helm_manager: HelmManager,
        alert_manager: AlertManager,
        prometheus_url: str = "http://prometheus:9090"
    ):
        """Initialize canary validator.
        
        Args:
            helm_manager: Helm deployment manager
            alert_manager: Alert notification manager
            prometheus_url: Prometheus server URL
        """
        self.helm = helm_manager
        self.alerts = alert_manager
        self.analyzer = CanaryAnalyzer(prometheus_url=prometheus_url)
        self.policy = RollbackPolicy()
    
    async def validate_canary(
        self,
        version: str,
        production_version: str,
        values_file: str,
        wait_time_minutes: int = 10
    ) -> bool:
        """Run complete canary validation process.
        
        Args:
            version: New version to validate
            production_version: Current production version
            values_file: Helm values file
            wait_time_minutes: Time to wait before validation
            
        Returns:
            True if validation passed
        """
        logger.info(f"Starting canary validation for {version}")
        
        try:
            # 1. Deploy canary
            if not await self.helm.deploy_canary(version, values_file):
                await self.alerts.send_alert(
                    f"❌ Failed to deploy canary {version}"
                )
                return False
            
            # 2. Wait for metrics to accumulate
            logger.info(f"Waiting {wait_time_minutes} minutes for metrics...")
            await asyncio.sleep(wait_time_minutes * 60)
            
            # 3. Fetch and compare metrics
            logger.info("Fetching metrics for comparison...")
            decision, reasoning = await evaluate_canary(
                production_version,
                f"{version}-canary",
                duration_minutes=wait_time_minutes
            )
            
            # 4. Make decision based on metrics
            if decision == RollbackDecision.ROLLBACK_AUTO:
                logger.warning("Auto-rollback triggered")
                await self.helm.rollback()
                await self.alerts.notify_rollback(version, reasoning["primary_reason"])
                return False
            
            elif decision == RollbackDecision.PROCEED:
                logger.info("Canary validation passed")
                await self.helm.promote_canary()
                await self.alerts.notify_success(version)
                return True
            
            else:  # ROLLBACK_MANUAL
                logger.warning("Manual review required")
                await self.alerts.notify_manual_review(version, reasoning)
                
                # In CI/CD, we might want to fail the pipeline
                # or wait for manual approval
                return False
        
        except Exception as e:
            logger.error(f"Canary validation failed: {e}")
            await self.helm.rollback()
            await self.alerts.send_alert(
                f"❌ Canary validation failed for {version}: {str(e)}"
            )
            return False
        
        finally:
            await self.analyzer.close()


# GitHub Actions integration
async def github_actions_canary_validation():
    """Main function for GitHub Actions integration.
    
    Expected environment variables:
    - VERSION: Version being deployed
    - PRODUCTION_VERSION: Current production version
    - VALUES_FILE: Path to Helm values file
    - SLACK_WEBHOOK_URL: Optional Slack webhook
    - PROMETHEUS_URL: Prometheus server URL
    """
    # Get configuration from environment
    version = os.getenv("VERSION", "latest")
    production_version = os.getenv("PRODUCTION_VERSION", "v1.0.0")
    values_file = os.getenv("VALUES_FILE", "values.yaml")
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    
    # Initialize components
    helm = HelmManager()
    alerts = AlertManager(slack_webhook)
    validator = CanaryValidator(helm, alerts, prometheus_url)
    
    # Run validation
    success = await validator.validate_canary(
        version=version,
        production_version=production_version,
        values_file=values_file,
        wait_time_minutes=15
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


# Example usage
async def main():
    """Example of running canary validation."""
    import os
    
    # Configuration
    config = {
        "version": "v1.3.0",
        "production_version": "v1.2.0",
        "values_file": "charts/apex-inference/values-canary.yaml",
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
        "prometheus_url": "http://prometheus:9090"
    }
    
    # Run validation
    helm = HelmManager()
    alerts = AlertManager(config["slack_webhook"])
    validator = CanaryValidator(helm, alerts, config["prometheus_url"])
    
    success = await validator.validate_canary(
        version=config["version"],
        production_version=config["production_version"],
        values_file=config["values_file"],
        wait_time_minutes=10
    )
    
    if success:
        print("✅ Canary validation passed!")
    else:
        print("❌ Canary validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    """Run canary validation example."""
    import os
    
    print("=" * 60)
    print("🚀 Canary Validation Example")
    print("=" * 60)
    
    # Check if running in CI
    if os.getenv("CI"):
        asyncio.run(github_actions_canary_validation())
    else:
        asyncio.run(main())
    
    print("\nNext steps for CI/CD integration:")
    print("1. Add to .github/workflows/deploy.yml")
    print("2. Configure environment variables")
    print("3. Set up Slack webhook for alerts")
    print("4. Configure Prometheus queries")
    print("5. Test with staging environment first")
