tell application "Google Chrome"
activate
delay 0.5
end tell
tell application "System Events"
tell application process "Google Chrome"
set myList to {"gcc", "python", "grep", "ls", "emacs", "pip install", "apt"}
repeat
set s to some item of myList
repeat with x in s
key code 102
keystroke x
delay ((random number) + 0.1)
end repeat
delay 0.5
repeat with x in s
key code 51 using {command down}
delay 0.5
end repeat
delay 0.5
key code 36
delay (random number from 60 to 120)
end repeat
end tell
end tell