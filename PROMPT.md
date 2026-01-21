<!--
  PROMPT.md - Instructions for the ralph.sh automation loop

  This file is read by ralph.sh and passed to Claude Code via:
    claude -p "$(cat PROMPT.md)" --output-format text

  The @file references at the top tell Claude to read those files as context.

  CHANGE LOG:
  - 2026-01-20: Removed "Start the server locally" instruction that was causing
    300s timeouts. The server runs indefinitely, blocking the automation loop.
    Tests can run without the server being started first.
-->

<!-- Context files - Claude will read these automatically -->
@plan.md @activity.md

<!-- Project context -->
We are rebuilding the project from scratch in this repo.

<!-- Step 1: Understand current state -->
First read activity.md to see what was recently accomplished.

<!-- Step 2: Pick next task -->
Open plan.md and choose the single highest priority task where passes is false.

<!-- Step 3: Implement (one task only to keep iterations focused) -->
Work on exactly ONE task: implement the change.

<!-- Step 4: Verify the implementation -->
After implementing, verify by running:
1. Run pytest on relevant tests
2. Save output to verification/[task-name].txt

<!-- Step 5: Update tracking files -->
Append a dated progress entry to activity.md describing what you changed and the verification filename.

Update that task's passes in plan.md from false to true.

<!-- Step 6: Commit (no push - user will review and push manually) -->
Make one git commit for that task only with a clear message.

Do not git init, do not change remotes, do not push.

<!-- Constraints -->
ONLY WORK ON A SINGLE TASK.

<!-- Completion signal - ralph.sh looks for this to exit the loop -->
When ALL tasks have passes true, output <promise>COMPLETE</promise>
