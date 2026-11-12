#!/bin/sh
# Render the unit templates into a directory (default: ./rendered). Review, then copy to
# /etc/systemd/system yourself. Nothing here touches /etc or starts a service.
#
#   AAR_ROOT=/var/lib/aar AAR_USER=aar AAR_GROUP=aar TAILNET_IP=100.x.y.z \
#   CONSOLE_HOSTNAME=host.tailnet.ts.net AAR_MEDIA_ROOT=/srv/aar-media \
#   AAR_WORK_DIR=/srv/aar-media/work AAR_EXPORT_DIR=/srv/aar-media/share/exports \
#   ./install-units.sh [outdir]
#
# It also renders hub.env (the units' EnvironmentFile): copy it to $AAR_ROOT/hub.env.
# The media, work and export paths may hold spaces and any of | & / (they are quoted in the
# units and in hub.env); a value with a double quote, a backslash, % or $ is refused, because
# systemd would read those as quoting, specifiers or variables. AAR_ROOT, the user, the group,
# the address and the host name may not hold whitespace either.
set -eu
: "${AAR_ROOT:?}" "${AAR_USER:?}" "${AAR_GROUP:?}" "${TAILNET_IP:?}" "${CONSOLE_HOSTNAME:?}"
: "${AAR_MEDIA_ROOT:?}" "${AAR_WORK_DIR:?}" "${AAR_EXPORT_DIR:?}"
nl='
'
for name in AAR_ROOT AAR_USER AAR_GROUP TAILNET_IP CONSOLE_HOSTNAME \
            AAR_MEDIA_ROOT AAR_WORK_DIR AAR_EXPORT_DIR; do
  eval "value=\${$name}"
  case $value in
    *\"*|*\\*|*%*|*\$*|*"$nl"*)
      echo "refused: $name holds a double quote, backslash, %, \$ or newline" >&2; exit 2 ;;
  esac
done
for name in AAR_ROOT AAR_USER AAR_GROUP TAILNET_IP CONSOLE_HOSTNAME; do
  eval "value=\${$name}"
  case $value in
    *[[:space:]]*) echo "refused: $name holds whitespace" >&2; exit 2 ;;
  esac
done
# A value as a sed replacement: escape the delimiter |, & and \.
esc() { printf '%s' "$1" | sed -e 's/[\\|&]/\\&/g'; }
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-./rendered}
mkdir -p "$out"
for t in "$here"/*.service.in "$here"/*.env.in; do
  name=$(basename "$t" .in)
  sed -e "s|@AAR_ROOT@|$(esc "$AAR_ROOT")|g" -e "s|@AAR_USER@|$(esc "$AAR_USER")|g" \
      -e "s|@AAR_GROUP@|$(esc "$AAR_GROUP")|g" -e "s|@TAILNET_IP@|$(esc "$TAILNET_IP")|g" \
      -e "s|@CONSOLE_HOSTNAME@|$(esc "$CONSOLE_HOSTNAME")|g" \
      -e "s|@AAR_MEDIA_ROOT@|$(esc "$AAR_MEDIA_ROOT")|g" \
      -e "s|@AAR_WORK_DIR@|$(esc "$AAR_WORK_DIR")|g" \
      -e "s|@AAR_EXPORT_DIR@|$(esc "$AAR_EXPORT_DIR")|g" \
      "$t" > "$out/$name"
  echo "wrote $out/$name"
done
