# News Scraper for BETenio

A simple, ethical web scraper for downloading and extracting text content from news articles.

## Features

- ✅ **Respects robots.txt** - Automatically checks and honors site policies
- ✅ **Polite delays** - Configurable delays (8s base + up to 3s random jitter) between requests
- ✅ **Retry logic** - Automatic retries with exponential backoff on failures
- ✅ **Smart extraction** - Uses Lynx for clean text extraction, with fallback to basic HTML cleaning
- ✅ **Organized output** - Saves both HTML and plain text in separate folders
- ✅ **Detailed logging** - Complete log of all operations for debugging
- ✅ **Handles errors gracefully** - Continues processing even if individual URLs fail

## Requirements

### Linux/WSL (Ubuntu)

```bash
# Essential (required)
sudo apt-get update
sudo apt-get install curl

# Recommended (for better text extraction)
sudo apt-get install lynx
```

### macOS

```bash
# Using Homebrew
brew install curl lynx
```

### Windows

Use **WSL (Windows Subsystem for Linux)**:

1. Open PowerShell as Administrator and run:

   ```powershell
   wsl --install
   ```

2. Restart your computer

3. Open "Ubuntu" from Start menu

4. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install curl lynx
   ```

## Installation

1. Download the script:

   ```bash
   wget https://github.com/andresquesadag/BETenio/news_scraper/ns.sh
   # or
   curl -O https://github.com/andresquesadag/BETenio/news_scraper/ns.sh
   ```

2. Make it executable:
   ```bash
   chmod +x ns.sh
   ```

## Usage

### Basic Usage

1. Create a file with URLs (one per line):

   ```bash
   cat > urls.txt << 'EOF'
   https://example.com/article1
   https://example.com/article2
   https://example.com/article3
   EOF
   ```

2. Run the scraper:
   ```bash
   ./ns.sh urls.txt
   ```

### Output Structure

```plain
news_data/
├── html/
│   ├── example_com_article1.html
│   ├── example_com_article2.html
│   └── example_com_article3.html
└── text/
    ├── example_com_article1.txt
    ├── example_com_article2.txt
    └── example_com_article3.txt
```

### Advanced Usage

#### Custom Configuration

```bash
# Change output directory
OUTDIR="my_articles" ./ns.sh urls.txt

# Increase delay between requests (more polite)
DELAY=15 ./ns.sh urls.txt

# Reduce jitter
JITTER=1 ./ns.sh urls.txt

# Combine multiple options
UA="MyBot/1.0 (+myemail@domain.com)" DELAY=12 OUTDIR="data" ./ns.sh urls.txt
```

#### Configuration Options

| Variable      | Default                                 | Description                              |
| ------------- | --------------------------------------- | ---------------------------------------- |
| `OUTDIR`      | `news_data`                             | Output directory for scraped content     |
| `LOGFILE`     | `scraper.log`                           | Log file path                            |
| `UA`          | `NewsBot/1.0 (+your-email@example.com)` | User-Agent string                        |
| `DELAY`       | `8`                                     | Base delay in seconds between requests   |
| `JITTER`      | `3`                                     | Maximum random additional seconds to add |
| `MAX_RETRIES` | `3`                                     | Number of retry attempts on failure      |

### URL File Format

```txt
# Lines starting with # are ignored (comments)
https://site1.com/article1

# Empty lines are ignored

https://site2.com/article2
https://site3.com/article3
```

## Examples

### Process a Small Batch

```bash
# Create a small test batch
cat > test_urls.txt << 'EOF'
https://news-site.com/article1
https://news-site.com/article2
EOF

# Run with default settings
./ns.sh test_urls.txt
```

### Process Multiple Batches with Breaks

```bash
# Process different news sources separately
./ns.sh source1_urls.txt
sleep 300  # Wait 5 minutes

./ns.sh source2_urls.txt
sleep 300

./ns.sh source3_urls.txt
```

### Use a Real User-Agent

```bash
# Always use a real email for ethical scraping
UA="ResearchBot/1.0 (+myemail@university.edu)" ./ns.sh urls.txt
```

## Ethical Scraping Guidelines

This script follows web scraping best practices:

- ✅ **Respects robots.txt** - Won't scrape if disallowed
- ✅ **Rate limiting** - Waits between requests to avoid overloading servers
- ✅ **Identifies itself** - Uses a clear User-Agent (customize with your email!)
- ✅ **Handles errors** - Doesn't hammer servers on failures
- ✅ **Transparent** - Logs all actions

### Important Notes

- **Always use your real email** in the User-Agent
- **Start with small batches** to test (10-20 URLs)
- **Check Terms of Service** of each site before scraping
- **Don't bypass paywalls** or authentication
- **Be patient** - The delays exist for a reason
- **Use for research/personal use** - Not for republishing content

## Troubleshooting

### "curl: command not found"

Install curl:

```bash
sudo apt-get install curl
```

### "Permission denied"

Make the script executable:

```bash
chmod +x ns.sh
```

### "Failed to download" errors

- **Check your internet connection**
- **Verify the URL is accessible** (try opening in browser)
- **Site may be blocking bots** - Check their robots.txt
- **Try increasing delay**: `DELAY=15 ./ns.sh urls.txt`

### Empty text files but HTML downloaded

- **Install lynx** for better text extraction:
  ```bash
  sudo apt-get install lynx
  ```
- The script will fall back to basic HTML stripping without lynx

### Script stops after first URL

- **Check URL file format** - Ensure Unix line endings:
  ```bash
  dos2unix urls.txt
  # or
  sed -i 's/\r$//' urls.txt
  ```

## Logs

All operations are logged to `scraper.log`:

```bash
# View logs in real-time
tail -f scraper.log

# Search for errors
grep "ERROR" scraper.log

# Count successful downloads
grep "SUCCESS" scraper.log | wc -l
```

## Performance

- **Speed**: ~8-11 seconds per URL (due to polite delays)
- **For 100 URLs**: ~15-20 minutes
- **For 1000 URLs**: ~2.5-3 hours

These times are **by design** to be respectful to servers.

## License

Use responsibly and ethically. Always respect website Terms of Service and robots.txt.

## Support

For issues or questions, check:

- The log file: `scraper.log`
- robots.txt of the target site
- Your network connection
- File permissions
