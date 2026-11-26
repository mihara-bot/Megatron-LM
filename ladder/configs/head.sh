# LM head + token margin (SphereFace-R, megatron/core/models/gpt/token_margin.py). Sourced by
# ladder/train.sh, ladder/sweep.sh and ladder/export/export_hf.sh; reads the environment:
#
#   HEAD            plain (default) | nfn | hfn | sfn
#                     nfn: logits ||h|| cos(h, W_i); hfn: s cos(h, W_i), final norm weight frozen at 1;
#                     sfn: nfn logits + t (||h|| - s)^2 per token in the training loss
#   MARGIN_D        cosine margin d >= 0 on the target token (default 0 = feature-norm-only control).
#                   Logit margin m = d s (hfn) or d ||h|| (nfn, sfn). Training loss only.
#   MARGIN_REVERSE  1: cos + d instead of cos - d (needs MARGIN_D > 0)
#   HEAD_SCALE      s: hfn feature norm / sfn penalty target (default 60)
#   SFN_WEIGHT      t: sfn penalty weight (default 1e-3)
#
# Sets HEAD_ARGS (Megatron flags) and HEAD_TAG (run-name suffix, empty for the plain head, so the
# existing baseline run directories keep their names). The head does not change the data order:
# runs that differ only in the head read the same batches, so their validation losses pair.
HEAD=${HEAD:-plain}
MARGIN_D=${MARGIN_D:-0}
MARGIN_REVERSE=${MARGIN_REVERSE:-0}
HEAD_SCALE=${HEAD_SCALE:-60}
SFN_WEIGHT=${SFN_WEIGHT:-1e-3}

_is_nonneg() { awk -v x="$1" 'BEGIN { exit !(x ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) }'; }
_is_zero() { awk -v x="$1" 'BEGIN { exit !(x + 0 == 0) }'; }
for _v in MARGIN_D HEAD_SCALE SFN_WEIGHT; do
    if ! _is_nonneg "${!_v}"; then echo "$_v=${!_v} must be a non-negative number" >&2; exit 1; fi
done
case "$MARGIN_REVERSE" in 0|1) ;; *) echo "MARGIN_REVERSE must be 0 or 1" >&2; exit 1 ;; esac

HEAD_ARGS=()
HEAD_TAG=""
case "$HEAD" in
plain)
    if ! _is_zero "$MARGIN_D" || [ "$MARGIN_REVERSE" = 1 ]; then
        echo "MARGIN_D / MARGIN_REVERSE need HEAD=nfn|hfn|sfn" >&2; exit 1
    fi
    ;;
nfn|hfn|sfn)
    if _is_zero "$HEAD_SCALE"; then echo "HEAD_SCALE must be > 0" >&2; exit 1; fi
    if [ "$MARGIN_REVERSE" = 1 ] && _is_zero "$MARGIN_D"; then
        echo "MARGIN_REVERSE=1 needs MARGIN_D > 0" >&2; exit 1
    fi
    HEAD_ARGS=(--lm-head-feature-norm "$HEAD" --token-margin "$MARGIN_D")
    HEAD_TAG="_${HEAD}"
    if [ "$MARGIN_REVERSE" = 1 ]; then HEAD_ARGS+=(--token-margin-reverse); HEAD_TAG+="_rev"; fi
    HEAD_TAG+="_d${MARGIN_D}"
    if [ "$HEAD" != nfn ]; then
        HEAD_ARGS+=(--lm-head-feature-norm-scale "$HEAD_SCALE")
        if [ "$HEAD_SCALE" != 60 ]; then HEAD_TAG+="_s${HEAD_SCALE}"; fi
    fi
    if [ "$HEAD" = sfn ]; then
        HEAD_ARGS+=(--soft-feature-norm-weight "$SFN_WEIGHT")
        if [ "$SFN_WEIGHT" != 1e-3 ]; then HEAD_TAG+="_t${SFN_WEIGHT}"; fi
    fi
    ;;
*)
    echo "HEAD must be plain, nfn, hfn or sfn, got $HEAD" >&2; exit 1 ;;
esac
unset _v
