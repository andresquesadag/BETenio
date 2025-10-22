# Copyright (2025) Andres Quesada G., Andre Salas, Felipe Bianchini

#!/usr/bin/env bash
# Simple news scraper: downloads HTML and extracts text content
# Requirements: curl, lynx (optional for better text extraction)
# Note: Respects robots.txt and includes polite delays

set -euo pipefail

# Configuration
URLS="${1:-urls.txt}"
OUTDIR="${OUTDIR:-news_data}"
LOGFILE="${LOGFILE:-scraper.log}"
UA="${UA:-News Scraper for BETenio/1.0 (+github.com/andresquesadag/BETenio)}"

DELAY="${DELAY:-8}"
JITTER="${JITTER:-3}"
MAX_RETRIES="${MAX_RETRIES:-3}"

mkdir -p "$OUTDIR/html" "$OUTDIR/text"

# Check dependencies
if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required"; exit 1
fi

HAVE_LYNX=0
command -v lynx >/dev/null 2>&1 && HAVE_LYNX=1

if [[ ! -f "$URLS" ]]; then
  echo "Error: Create $URLS with one URL per line"; exit 1
fi

# Utilities
log() {
  echo "[$(date -u +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOGFILE" >&2
}

safe_filename() {
  echo -n "$1" | sed -E 's#[^A-Za-z0-9._-]+#_#g'
}

check_robots() {
  local domain="$1"
  local tmp
  tmp=$(mktemp)
  
  if ! curl -sS --fail --max-time 10 -A "$UA" \
      "https://$domain/robots.txt" -o "$tmp" 2>/dev/null; then
    rm -f "$tmp"
    return 0  # No robots.txt, OK to scrape
  fi
  
  # Simple check for "User-agent: *" with "Disallow: /"
  if awk '
    BEGIN { capture=0; disallow_all=0 }
    tolower($0) ~ /^user-agent:[[:space:]]*\*/ { capture=1; next }
    tolower($0) ~ /^user-agent:/ { capture=0 }
    capture && tolower($0) ~ /^disallow:[[:space:]]*\/[[:space:]]*$/ { disallow_all=1 }
    END { exit(disallow_all ? 1 : 0) }
  ' "$tmp"; then
    rm -f "$tmp"
    return 0  # OK
  else
    rm -f "$tmp"
    return 1  # Disallowed
  fi
}

sleep_with_jitter() {
  local base="$1"
  local jitter=$((RANDOM % (JITTER + 1)))
  local total=$((base + jitter))
  log "Waiting ${total}s"
  sleep "$total"
}

# Download and extract
process_url() {
  local url="$1"
  local domain
  domain=$(echo "$url" | awk -F/ '{print $3}')
  
  local fname
  fname=$(safe_filename "$url")
  local html_file="$OUTDIR/html/${fname}.html"
  local text_file="$OUTDIR/text/${fname}.txt"
  
  # Check robots.txt
  if ! check_robots "$domain"; then
    log "SKIP: robots.txt blocks $domain"
    return 2
  fi
  
  # Download
  local attempt=0
  local success=0
  
  while ((attempt < MAX_RETRIES && success == 0)); do
    attempt=$((attempt + 1))
    sleep_with_jitter "$DELAY"
    
    log "Downloading (attempt $attempt/$MAX_RETRIES): $url"
    
    if curl -sS --fail --location --max-time 45 \
            -A "$UA" -o "$html_file" "$url" 2>/dev/null; then
      if [[ -s "$html_file" ]]; then
        success=1
        log "SUCCESS: Downloaded $(wc -c < "$html_file") bytes"
      else
        log "FAILED: Empty response"
        rm -f "$html_file"
      fi
    else
      log "FAILED: curl error"
      if ((attempt < MAX_RETRIES)); then
        sleep $((2 ** attempt))
      fi
    fi
  done
  
  if ((success == 0)); then
    log "ERROR: Failed to download after $MAX_RETRIES attempts"
    return 1
  fi
  
  # Extract text
  if ((HAVE_LYNX == 1)); then
    if lynx -dump -nolist -stdin < "$html_file" > "$text_file" 2>/dev/null; then
      log "SUCCESS: Text extracted with lynx -> $text_file"
      return 0
    fi
  fi
  
  # Fallback: basic HTML cleaning
  sed -e 's/<script[^>]*>.*<\/script>//gi' \
      -e 's/<style[^>]*>.*<\/style>//gi' \
      -e 's/<[^>]*>//g' \
      -e 's/&nbsp;/ /g' \
      -e 's/&amp;/\&/g' \
      -e 's/&lt;/</g' \
      -e 's/&gt;/>/g' \
      -e 's/&quot;/"/g' \
      "$html_file" | \
      sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | \
      grep -v '^$' > "$text_file" || true
  
  if [[ -s "$text_file" ]]; then
    log "SUCCESS: Text extracted (basic) -> $text_file"
    return 0
  else
    log "WARNING: Could not extract text, but HTML saved"
    return 0
  fi
}

# Main
log "Starting scraper..."
log "Input: $URLS"
log "Output: $OUTDIR/"
log "Lynx: $HAVE_LYNX"

processed=0
failed=0

# Read file line by line, handling different line endings
while IFS= read -r url || [[ -n "$url" ]]; do
  # Remove carriage returns and whitespace
  url=$(echo "$url" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  
  # Skip empty lines and comments
  [[ -z "$url" ]] && continue
  [[ "$url" =~ ^# ]] && continue
  
  log "===== Processing: $url ====="
  
  if process_url "$url"; then
    processed=$((processed + 1))
    log "Status: SUCCESS"
  else
    failed=$((failed + 1))
    log "Status: FAILED"
  fi
  
  log ""
  
done < "$URLS"

log "===== Finished ====="
log "Processed: $processed | Failed: $failed"
log "Files saved in: $OUTDIR/"