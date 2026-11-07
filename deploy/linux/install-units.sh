#!/bin/sh
# Render the unit templates into a directory (default: ./rendered). Review, then copy to
# /etc/systemd/system yourself. Nothing here touches /etc or starts a service.
#
#   AAR_ROOT=/var/lib/aar AAR_USER=aar AAR_GROUP=aar TAILNET_IP=100.x.y.z \
#   CONSOLE_HOSTNAME=host.tailnet.ts.net ./install-units.sh [outdir]
set -eu
: "${AAR_ROOT:?}" "${AAR_USER:?}" "${AAR_GROUP:?}" "${TAILNET_IP:?}" "${CONSOLE_HOSTNAME:?}"
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-./rendered}
mkdir -p "$out"
for t in "$here"/*.service.in; do
  name=$(basename "$t" .in)
  sed -e "s|@AAR_ROOT@|$AAR_ROOT|g" -e "s|@AAR_USER@|$AAR_USER|g" -e "s|@AAR_GROUP@|$AAR_GROUP|g" \
      -e "s|@TAILNET_IP@|$TAILNET_IP|g" -e "s|@CONSOLE_HOSTNAME@|$CONSOLE_HOSTNAME|g" \
      "$t" > "$out/$name"
  echo "wrote $out/$name"
done
