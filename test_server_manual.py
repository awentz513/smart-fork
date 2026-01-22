#!/usr/bin/env python3
"""Quick manual test of the MCP server"""

import sys
import json
import subprocess

def test_server():
    """Test if the server responds to basic requests"""

    # Test 1: List tools
    print("Test 1: Listing available tools...")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }

    try:
        result = subprocess.run(
            [sys.executable, "-m", "smart_fork.server"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout:
            response = json.loads(result.stdout)
            if "result" in response:
                tools = response["result"].get("tools", [])
                print(f"✅ Server responded! Found {len(tools)} tools:")
                for tool in tools[:5]:  # Show first 5
                    print(f"   - {tool.get('name')}")
                return True
            else:
                print(f"❌ Unexpected response format: {response}")
                return False
        else:
            print(f"❌ Server failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Server timed out (took > 10s)")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        print(f"stdout: {result.stdout}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Smart Fork MCP Server...\n")
    success = test_server()

    if success:
        print("\n✅ Basic server functionality works!")
        print("\nNext: Try calling fork-detect with a real query")
    else:
        print("\n❌ Server is not working properly")
        print("Check the error messages above")

    sys.exit(0 if success else 1)
