#!/bin/bash

# ralph3.sh - Phase 3 automation loop for feature development
# Uses PROMPT3.md, plan3.md, activity3.md

# Exit on undefined variables
set -u

# Detect timeout command (gtimeout on macOS with coreutils, timeout on Linux)
if command -v gtimeout &> /dev/null; then
  TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
  TIMEOUT_CMD="timeout"
else
  TIMEOUT_CMD=""
fi

# Configuration
TIMEOUT_SECONDS=600  # 10 minute timeout per claude call (longer due to Smart Fork searches)
MAX_RETRIES=3
RETRY_DELAY=5
LOG_FILE="ralph_log_3.txt"
PROMPT_FILE="PROMPT3.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Cleanup function for graceful shutdown
cleanup() {
  echo -e "\n${YELLOW}[$(date '+%H:%M:%S')] Caught interrupt signal, cleaning up...${NC}"
  kill_orphan_servers
  exit 130
}

# Kill orphaned servers from previous iterations
kill_orphan_servers() {
  # Find and kill any smart_fork or uvicorn processes
  for pattern in "smart_fork" "uvicorn"; do
    local pids=$(pgrep -f "$pattern" 2>/dev/null)
    if [ -n "$pids" ]; then
      echo "$pids" | xargs kill 2>/dev/null
      sleep 0.5
      # Force kill if still running
      pids=$(pgrep -f "$pattern" 2>/dev/null)
      if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null
        log "Force-killed stubborn $pattern processes"
      else
        log "Killed orphaned $pattern processes"
      fi
    fi
  done

  # Clear MCP server port (8741) and fallback ports
  for port in 8741 8742 8743; do
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
      kill -9 $pid 2>/dev/null && log "Cleared port $port (killed PID $pid)"
    fi
  done
}

# Verify a port is truly available
verify_port_available() {
  local port=$1
  local max_attempts=5

  for ((i=1; i<=max_attempts; i++)); do
    if ! lsof -ti:$port > /dev/null 2>&1; then
      return 0  # Port is free
    fi

    if curl -s --max-time 1 http://localhost:$port/ > /dev/null 2>&1; then
      return 1  # Port is in use by a working server
    fi

    log "${YELLOW}Port $port bound but unresponsive, attempting cleanup...${NC}"
    local pid=$(lsof -ti:$port 2>/dev/null)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null
    sleep 0.5
  done

  return 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

log() {
  local msg="[$(date '+%H:%M:%S')] $1"
  echo -e "$msg"
  echo "$msg" >> "$LOG_FILE"
}

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <iterations>"
  echo ""
  echo "Phase 3 automation loop for feature development."
  echo "Uses: $PROMPT_FILE, plan3.md, activity3.md"
  echo "Logs to: $LOG_FILE"
  exit 1
fi

# Check required files exist
if [ ! -f "$PROMPT_FILE" ]; then
  echo -e "${RED}Error: $PROMPT_FILE not found${NC}"
  exit 1
fi

if [ ! -f "plan3.md" ]; then
  echo -e "${RED}Error: plan3.md not found${NC}"
  exit 1
fi

if [ ! -f "activity3.md" ]; then
  echo -e "${RED}Error: activity3.md not found${NC}"
  exit 1
fi

# Create verification directory if needed
mkdir -p verification

# Append to log file (don't overwrite)
echo "" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"
echo "Ralph Phase 3 loop started at $(date)" >> "$LOG_FILE"
log "${CYAN}=== PHASE 3: Feature Development ===${NC}"

# Warn if no timeout command available
if [ -z "$TIMEOUT_CMD" ]; then
  log "${YELLOW}Warning: No timeout command found. Install coreutils (brew install coreutils) for timeout support.${NC}"
else
  log "Using $TIMEOUT_CMD for ${TIMEOUT_SECONDS}s timeout"
fi

# Clean up orphaned servers from previous runs
kill_orphan_servers

for ((i=1; i<=$1; i++)); do
  log "${GREEN}Phase 3 - Iteration $i of $1${NC}"
  log "--------------------------------"

  # Retry loop for transient failures
  for ((retry=1; retry<=MAX_RETRIES; retry++)); do
    log "Attempt $retry/$MAX_RETRIES..."

    # Run claude with timeout (if available)
    if [ -n "$TIMEOUT_CMD" ]; then
      result=$($TIMEOUT_CMD "$TIMEOUT_SECONDS" claude -p "$(cat $PROMPT_FILE)" --output-format text 2>&1)
      exit_code=$?
    else
      result=$(claude -p "$(cat $PROMPT_FILE)" --output-format text 2>&1)
      exit_code=$?
    fi

    # Check if command succeeded
    if [ $exit_code -eq 0 ]; then
      break  # Success, exit retry loop
    fi

    # Handle failure
    if [ $exit_code -eq 124 ]; then
      log "${RED}Claude command timed out after ${TIMEOUT_SECONDS}s${NC}"
    else
      log "${RED}Claude command failed with exit code $exit_code${NC}"
    fi

    if [ $retry -lt $MAX_RETRIES ]; then
      log "${YELLOW}Retrying in ${RETRY_DELAY}s...${NC}"
      sleep $RETRY_DELAY
    else
      log "${RED}Max retries reached, continuing to next iteration${NC}"
      result="[ERROR: All retries failed]"
    fi
  done

  echo "$result"
  echo "$result" >> "$LOG_FILE"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    log "${GREEN}=== PHASE 3 COMPLETE ===${NC}"
    log "${GREEN}All feature development tasks complete after $i iterations.${NC}"
    exit 0
  fi

  log "--- End of Phase 3 iteration $i ---"
  echo ""
done

log "${YELLOW}Reached max iterations ($1)${NC}"
exit 1
