# Value Investing Analysis Skill

## One-line Summary
Activate this skill when the user mentions stock tickers, companies such as Tencent, PDD, Apple, or sectors like consumer goods, healthcare, liquor, or the Hang Seng Tech Index.

## Trigger Conditions
Activate this skill when:
- The user mentions a stock ticker (e.g. 600519, HK00700, AAPL).
- The user asks questions like "What do you think about this stock?" or "Can you analyze this company?"
- The conversation involves companies or sectors that Wang Qian frequently follows, such as Tencent, PDD, Hang Seng Tech, liquor, healthcare, consumer goods, and similar investments.

## Analysis Workflow

### STEP 1: Retrieve Market Data
Call `stock_query(stock_code)` to obtain real-time market information:

- Current price
- Price change (%)
- Trading volume
- Turnover rate
- Price-to-Earnings ratio (PE)
- Price-to-Book ratio (PB)
- Market capitalization
- 52-week high / low (for Hong Kong stocks)

### STEP 2: Search Recent News
Call `web_search(query)` using queries such as:

- "{Company Name} latest news"
- "{Company Name} earnings" or "{Company Name} financial results"

### STEP 3: Comprehensive Analysis

### Fundamentals (Weight: 50%)
- PE < 15 → Potentially undervalued.
- PE > 40 → Potentially overvalued.
- PB < 1.5 → Strong margin of safety.
- Evaluate whether the company's market capitalization is reasonable compared to industry leaders.

### Technical Analysis (Weight: 30%)
- Price increase > +5% → Avoid chasing momentum.
- Price decrease < -5% → Possible panic selling; observe carefully.
- Turnover rate > 10% → Active trading; pay attention to risk.
- Trading near the 52-week low → May present a buying opportunity.

### News & Events (Weight: 20%)
- Evaluate how positive or negative news affects the company's fundamentals.
- Distinguish between one-time events and long-term trends.

## STEP 4: Output Rules
- Provide a clear recommendation:
  - Buy
  - Hold
  - Watch
  - Sell
- Include a target price range.
- Include a stop-loss price.
- Provide one sentence describing the main investment risk.
- End with a brief disclaimer.

## Mandatory Rules
- Never recommend an exact timing for buying or selling (e.g. "Buy at 2 PM").
- Keep the analysis of a single stock within three replies.
- Do not predict short-term price movements.