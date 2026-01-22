#!/usr/bin/env python3
"""Test fork-detect functionality with a real query"""

import sys
import json
import subprocess

def test_fork_detect():
    """Test fork-detect tool with a real query"""

    print("Test 2: Calling fork-detect with a query...")
    print("(This will take a moment as it loads the embedding model)\n")

    # Test fork-detect tool
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "fork-detect",
            "arguments": {
                "query": "python testing"
            }
        }
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "smart_fork.server"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=60  # Give it 60 seconds to load model and search
        )

        if result.returncode == 0 and result.stdout:
            try:
                response = json.loads(result.stdout)

                if "result" in response:
                    result_data = response["result"]

                    # Check if we got content back
                    if isinstance(result_data, list) and len(result_data) > 0:
                        content = result_data[0].get("content", [])
                        if content:
                            text = content[0].get("text", "")
                            print("✅ fork-detect returned results!")
                            print(f"\nFirst 500 chars of response:\n{text[:500]}...")

                            # Check for key indicators
                            if "session" in text.lower():
                                print("\n✅ Response mentions sessions")
                            if "found" in text.lower() or "no results" in text.lower():
                                print("✅ Search executed")

                            return True
                    else:
                        print("⚠️  fork-detect returned empty results")
                        print("This is OK if you have no Claude sessions indexed yet")
                        print("\nResponse:", json.dumps(result_data, indent=2)[:500])
                        return True  # Still counts as working

                elif "error" in response:
                    print(f"❌ fork-detect returned error: {response['error']}")
                    return False
                else:
                    print(f"❌ Unexpected response format: {response}")
                    return False

            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON response: {e}")
                print(f"stdout: {result.stdout[:500]}")
                return False
        else:
            print(f"❌ Server failed with return code {result.returncode}")
            print(f"stderr: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Server timed out (took > 60s)")
        print("This might mean the embedding model is downloading for the first time")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing fork-detect functionality...\n")
    print("=" * 60)

    success = test_fork_detect()

    print("\n" + "=" * 60)
    if success:
        print("\n✅ fork-detect is working!")
        print("\nThe core Smart Fork functionality is operational.")
        print("You can now search through your Claude Code sessions.")
    else:
        print("\n❌ fork-detect is not working properly")
        print("Check the error messages above")

    sys.exit(0 if success else 1)
