#!/usr/bin/env bash
# tools/crash_watch.sh PID: the GPU's state once a second into the systemd journal while the game
# (PID) runs and for 120 s after it ends (the PC has reset itself after the game was quit, the GPU
# lost from the bus first). Priority crit makes journald write each line to disk at once: a file's
# last ~30 s are lost in such a reset (btrfs commits every 30 s), the journal's last second is not.
# Read back after a reset: journalctl -b -1 -t bbport-watch
pid=${1:?usage: crash_watch.sh PID}
command -v systemd-cat >/dev/null || exit 0
card=""
for c in /sys/class/drm/card*/device; do
    [[ -r $c/gpu_busy_percent ]] && { card=$c; break; }
done
[[ -n $card ]] || exit 0
hw=$(ls -d "$card"/hwmon/hwmon* | head -1)
read_or() { cat "$1" 2>/dev/null || echo "$2"; }
after=0
while :; do
    if kill -0 "$pid" 2>/dev/null; then game=running; else game=ended; ((after++)); fi
    ((after > 120)) && break
    power=$(( $(read_or "$hw/power1_average" 0) / 1000000 ))
    printf 'game %s, %s W, busy %s%%, sclk %s MHz, edge %s C, junction %s C, mem %s C, fan %s rpm, link %s x%s\n' \
        "$game" "$power" "$(read_or "$card/gpu_busy_percent" ?)" \
        "$(( $(read_or "$hw/freq1_input" 0) / 1000000 ))" \
        "$(( $(read_or "$hw/temp1_input" 0) / 1000 ))" "$(( $(read_or "$hw/temp2_input" 0) / 1000 ))" \
        "$(( $(read_or "$hw/temp3_input" 0) / 1000 ))" "$(read_or "$hw/fan1_input" ?)" \
        "$(read_or "$card/current_link_speed" ?)" "$(read_or "$card/current_link_width" ?)"
    sleep 1
done | systemd-cat -t bbport-watch -p crit
