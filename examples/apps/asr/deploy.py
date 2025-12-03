"""
Deployment script for Flyte Speech Transcription Demo

This script deploys both the GPU transcriber service and the web frontend,
demonstrating Flyte's app-to-app calling pattern.
"""

import logging
import pathlib
import sys

# Import both apps
from transcriber import app as transcriber_app
from web_app import env as web_env

import flyte


def main():
    """Deploy both apps to Flyte."""
    print("=" * 70)
    print("🚀 Flyte Speech Transcription - Deployment")
    print("=" * 70)
    print()

    # Initialize Flyte
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )

    # Deploy transcriber service first (GPU backend)
    print("\n📦 Step 1/2: Deploying Parakeet Transcriber Service (GPU)...")
    print("-" * 70)
    try:
        transcriber_deployments = flyte.deploy(transcriber_app)

        if not transcriber_deployments:
            print("❌ Failed to deploy transcriber service")
            return 1

        print("✅ Transcriber service deployed successfully!")
        for d in transcriber_deployments:
            print(f"\n{d.table_repr()}")

    except Exception as e:
        print(f"❌ Error deploying transcriber: {e}")
        return 1

    # Deploy web frontend (CPU)
    print("\n📦 Step 2/2: Deploying Web Frontend (CPU)...")
    print("-" * 70)
    try:
        web_deployments = flyte.deploy(web_env)

        if not web_deployments:
            print("❌ Failed to deploy web frontend")
            return 1

        print("✅ Web frontend deployed successfully!")
        web_deployment = web_deployments[0]
        print(f"\n{web_deployment.table_repr()}")

        # Show access information
        print("\n" + "=" * 70)
        print("🎉 Deployment Complete!")
        print("=" * 70)
        print()
        print("📱 Access your application:")
        print(f"   Frontend URL:    {web_deployment.endpoint}/")
        print(f"   WebSocket:       {web_deployment.endpoint}/ws")
        print(f"   Health Check:    {web_deployment.endpoint}/health")
        print()
        print("🎤 To use the application:")
        print("   1. Open the frontend URL in your browser")
        print("   2. Click 'Connect' to establish WebSocket connection")
        print("   3. Allow microphone access when prompted")
        print("   4. Click 'Start Recording' to begin transcription")
        print("   5. Speak clearly into your microphone")
        print()
        print("💡 The frontend automatically calls the GPU transcriber service")
        print("   using Flyte's app-to-app calling pattern!")
        print()

        return 0

    except Exception as e:
        print(f"❌ Error deploying web frontend: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
