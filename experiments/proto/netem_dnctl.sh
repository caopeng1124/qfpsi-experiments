#!/bin/sh
# Kernel-level LAN/WAN emulation for the QFPSI prototype with dummynet (macOS).
#
# FLPSI's published settings (Uzun et al., USENIX Sec. 2021):
#   fast LAN: 500 MB/s, 0.5 ms RTT
#   slow WAN:  40 MB/s, 34 ms RTT
#
# Usage (needs root, run yourself):
#   sudo sh experiments/proto/netem_dnctl.sh fast     # 500 MB/s, 0.5 ms
#   sudo sh experiments/proto/netem_dnctl.sh slow     #  40 MB/s, 34 ms
#   sudo sh experiments/proto/netem_dnctl.sh off      # remove the rule
#
# Then run the benchmark against the shaped port, e.g.
#   .venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --port 9909
#
# Notes:
#  * dnctl bandwidth is in bit/s by default; "500Mbyte/s" also works on macOS.
#  * dummynet delay is one-way; RTT is therefore twice the configured value
#    when the rule is applied to a single interface. This script sets the
#    one-way delay to RTT/2 for that reason.
#  * pf must be enabled for the anchor to take effect on loopback traffic.

set -e

MODE="${1:-fast}"
ANCHOR="com.apple/qfpsi"
PORT="${2:-9909}"

case "$MODE" in
  fast) BW="500Mbyte/s"; DELAY="0.25ms" ;;
  slow) BW="40Mbyte/s";  DELAY="17ms"   ;;
  off)  BW=""           ; DELAY=""     ;;
  *) echo "usage: $0 {fast|slow|off} [port]"; exit 1 ;;
esac

if [ "$MODE" = "off" ]; then
  dnctl -q flush || true
  pfctl -q -a "$ANCHOR" -F rules || true
  echo "dummynet rules flushed"
  exit 0
fi

dnctl -q flush
dnctl pipe 1 config bw "$BW" delay "$DELAY"

cat > /tmp/qfpsi-pf.conf <<EOF
dummynet in quick on lo0 proto tcp from any to any port $PORT pipe 1
dummynet out quick on lo0 proto tcp from any port $PORT to any pipe 1
EOF

pfctl -q -a "$ANCHOR" -f /tmp/qfpsi-pf.conf
pfctl -e >/dev/null 2>&1 || true

echo "emulating $MODE network on loopback port $PORT: bw=$BW one-way delay=$DELAY (RTT approx $(echo "$DELAY" | sed 's/ms//')*2 ms)"
dnctl list
