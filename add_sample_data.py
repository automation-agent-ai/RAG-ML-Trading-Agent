#!/usr/bin/env python3
"""
Add sample data for testing the prediction pipeline

This script adds sample data for AFN_2023-11-20 to the DuckDB database,
including RNS announcements, news, fundamentals, analyst recommendations, and user posts.
"""

import os
import sys
import logging
import duckdb
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_rns_announcements(conn, setup_id="AFN_2023-11-20", ticker="AFN"):
    """Add sample RNS announcements"""
    logger.info(f"Adding sample RNS announcements for {setup_id}")
    
    # Check if table exists
    table_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'rns_announcements'
    """).fetchone()[0]
    
    if not table_exists:
        logger.info("Creating rns_announcements table")
        conn.execute("""
            CREATE TABLE rns_announcements (
                id INTEGER PRIMARY KEY,
                setup_id VARCHAR,
                ticker VARCHAR,
                headline VARCHAR,
                text VARCHAR,
                rns_date DATE,
                rns_time VARCHAR,
                url VARCHAR
            )
        """)
    
    # Sample RNS announcements
    sample_data = [
        {
            'id': 1001,
            'setup_id': setup_id,
            'ticker': ticker,
            'headline': f"{ticker} Reports Strong Q3 Financial Results",
            'text': f"{ticker} today announced strong financial results for the third quarter of 2023. Revenue increased by 15% year-over-year to £45.2 million, while net profit grew by 22% to £8.7 million. The company also reported improved operating margins of 19.3%, up from 17.8% in the same period last year. CEO John Smith commented: 'We are pleased with our performance this quarter, which reflects the success of our strategic initiatives and the dedication of our team. We remain confident in our ability to deliver sustainable growth and value for our shareholders.'",
            'rns_date': '2023-11-15',
            'rns_time': '07:00',
            'url': f"https://example.com/rns/{ticker}/2023-11-15"
        },
        {
            'id': 1002,
            'setup_id': setup_id,
            'ticker': ticker,
            'headline': f"{ticker} Announces New Contract Win",
            'text': f"{ticker} is pleased to announce that it has secured a significant new contract with a major client in the technology sector. The contract, valued at approximately £12 million over three years, involves the provision of the company's advanced software solutions and related services. This win represents an important expansion of the company's client base and reinforces its position as a leading provider in its market segment. The contract is expected to contribute to revenue from Q4 2023 onwards.",
            'rns_date': '2023-11-17',
            'rns_time': '09:30',
            'url': f"https://example.com/rns/{ticker}/2023-11-17"
        },
        {
            'id': 1003,
            'setup_id': setup_id,
            'ticker': ticker,
            'headline': f"{ticker} Board Changes",
            'text': f"{ticker} announces that Jane Wilson has been appointed as a Non-Executive Director with immediate effect. Jane brings over 25 years of experience in the software industry, having held senior positions at several leading technology companies. The company also announces that Robert Brown will be stepping down from the Board at the end of the year after serving for 8 years. The Chairman, David Johnson, commented: 'We are delighted to welcome Jane to the Board and look forward to benefiting from her extensive industry experience. We also thank Robert for his valuable contribution to the company over the years.'",
            'rns_date': '2023-11-18',
            'rns_time': '14:15',
            'url': f"https://example.com/rns/{ticker}/2023-11-18"
        }
    ]
    
    # Insert data
    for item in sample_data:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO rns_announcements
                (id, setup_id, ticker, headline, text, rns_date, rns_time, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                item['id'], 
                item['setup_id'], 
                item['ticker'], 
                item['headline'], 
                item['text'], 
                item['rns_date'], 
                item['rns_time'], 
                item['url']
            ])
        except Exception as e:
            logger.error(f"Error inserting RNS announcement: {e}")
    
    # Verify insertion
    count = conn.execute("SELECT COUNT(*) FROM rns_announcements WHERE setup_id = ?", [setup_id]).fetchone()[0]
    logger.info(f"Added {count} RNS announcements for {setup_id}")

def add_stock_news(conn, setup_id="AFN_2023-11-20", ticker="AFN"):
    """Add sample stock news"""
    logger.info(f"Adding sample stock news for {setup_id}")
    
    # Check if table exists
    table_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'stock_news_enhanced'
    """).fetchone()[0]
    
    if not table_exists:
        logger.info("Creating stock_news_enhanced table")
        conn.execute("""
            CREATE TABLE stock_news_enhanced (
                id INTEGER PRIMARY KEY,
                setup_id VARCHAR,
                ticker VARCHAR,
                title VARCHAR,
                content_summary VARCHAR,
                provider_publish_time TIMESTAMP,
                publisher VARCHAR,
                link VARCHAR
            )
        """)
    
    # Sample news
    sample_data = [
        {
            'id': 2001,
            'setup_id': setup_id,
            'ticker': ticker,
            'title': f"Analysts Upgrade {ticker} Following Strong Results",
            'content_summary': f"Following the release of strong Q3 results, several analysts have upgraded their outlook for {ticker}. Morgan Stanley raised its target price from £8.50 to £9.20, citing improved margins and better-than-expected revenue growth. Goldman Sachs maintained its 'Buy' rating and highlighted the company's robust cash generation. The stock has gained 5% since the results announcement, outperforming the broader market.",
            'provider_publish_time': '2023-11-16 10:45:00',
            'publisher': 'Financial Times',
            'link': f"https://example.com/ft/{ticker}/2023-11-16"
        },
        {
            'id': 2002,
            'setup_id': setup_id,
            'ticker': ticker,
            'title': f"{ticker}'s New Contract Could Boost Annual Revenue by 8%",
            'content_summary': f"Industry experts estimate that {ticker}'s recently announced contract win could boost its annual revenue by approximately 8%. The £12 million three-year deal with a major technology client is seen as a strategic win that could lead to further opportunities in the sector. The company's shares rose by 3% following the announcement, with trading volumes significantly above the three-month average.",
            'provider_publish_time': '2023-11-17 14:30:00',
            'publisher': 'Reuters',
            'link': f"https://example.com/reuters/{ticker}/2023-11-17"
        }
    ]
    
    # Insert data
    for item in sample_data:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO stock_news_enhanced
                (id, setup_id, ticker, title, content_summary, provider_publish_time, publisher, link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                item['id'], 
                item['setup_id'], 
                item['ticker'], 
                item['title'], 
                item['content_summary'], 
                item['provider_publish_time'], 
                item['publisher'], 
                item['link']
            ])
        except Exception as e:
            logger.error(f"Error inserting stock news: {e}")
    
    # Verify insertion
    count = conn.execute("SELECT COUNT(*) FROM stock_news_enhanced WHERE setup_id = ?", [setup_id]).fetchone()[0]
    logger.info(f"Added {count} stock news items for {setup_id}")

def add_fundamentals(conn, setup_id="AFN_2023-11-20", ticker="AFN"):
    """Add sample fundamentals data"""
    logger.info(f"Adding sample fundamentals for {setup_id}")
    
    # Check if fundamentals table exists
    fundamentals_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'fundamentals'
    """).fetchone()[0]
    
    if not fundamentals_exists:
        logger.info("Creating fundamentals table")
        conn.execute("""
            CREATE TABLE fundamentals (
                id INTEGER PRIMARY KEY,
                ticker VARCHAR,
                date DATE,
                total_revenue DOUBLE,
                gross_profit DOUBLE,
                operating_income DOUBLE,
                net_income DOUBLE,
                ebitda DOUBLE,
                operating_cash_flow DOUBLE,
                free_cash_flow DOUBLE,
                total_assets DOUBLE,
                total_debt DOUBLE,
                total_equity DOUBLE,
                current_assets DOUBLE,
                current_liabilities DOUBLE,
                working_capital DOUBLE
            )
        """)
    
    # Check if setups table exists
    setups_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'setups'
    """).fetchone()[0]
    
    if not setups_exists:
        logger.info("Creating setups table")
        conn.execute("""
            CREATE TABLE setups (
                setup_id VARCHAR PRIMARY KEY,
                lse_ticker VARCHAR,
                spike_timestamp TIMESTAMP
            )
        """)
        
        # Insert setup
        conn.execute("""
            INSERT INTO setups (setup_id, lse_ticker, spike_timestamp)
            VALUES (?, ?, ?)
        """, [setup_id, ticker, '2023-11-20 09:30:00'])
    else:
        # Check if setup exists
        setup_exists = conn.execute("""
            SELECT count(*) FROM setups 
            WHERE setup_id = ?
        """, [setup_id]).fetchone()[0]
        
        if not setup_exists:
            # Insert setup
            conn.execute("""
                INSERT INTO setups (setup_id, lse_ticker, spike_timestamp)
                VALUES (?, ?, ?)
            """, [setup_id, ticker, '2023-11-20 09:30:00'])
    
    # Check if fundamentals_features table exists
    features_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'fundamentals_features'
    """).fetchone()[0]
    
    if not features_exists:
        logger.info("Creating fundamentals_features table")
        conn.execute("""
            CREATE TABLE fundamentals_features (
                setup_id VARCHAR PRIMARY KEY,
                revenue_growth_yoy DOUBLE,
                profit_margin DOUBLE,
                debt_to_equity DOUBLE,
                current_ratio DOUBLE,
                return_on_equity DOUBLE,
                return_on_assets DOUBLE
            )
        """)
    
    # Sample fundamentals data
    sample_fundamentals = [
        {
            'id': 3001,
            'ticker': ticker,
            'date': '2023-09-30',  # Q3 2023
            'total_revenue': 45200000.0,
            'gross_profit': 18080000.0,
            'operating_income': 8729600.0,
            'net_income': 8700000.0,
            'ebitda': 10848000.0,
            'operating_cash_flow': 9040000.0,
            'free_cash_flow': 7684000.0,
            'total_assets': 120000000.0,
            'total_debt': 30000000.0,
            'total_equity': 70000000.0,
            'current_assets': 45000000.0,
            'current_liabilities': 25000000.0,
            'working_capital': 20000000.0
        },
        {
            'id': 3002,
            'ticker': ticker,
            'date': '2023-06-30',  # Q2 2023
            'total_revenue': 42500000.0,
            'gross_profit': 16575000.0,
            'operating_income': 7650000.0,
            'net_income': 7225000.0,
            'ebitda': 9775000.0,
            'operating_cash_flow': 8075000.0,
            'free_cash_flow': 6800000.0,
            'total_assets': 115000000.0,
            'total_debt': 32000000.0,
            'total_equity': 65000000.0,
            'current_assets': 42000000.0,
            'current_liabilities': 24000000.0,
            'working_capital': 18000000.0
        }
    ]
    
    # Sample fundamentals features
    sample_features = {
        'setup_id': setup_id,
        'revenue_growth_yoy': 0.15,  # 15% YoY growth
        'profit_margin': 0.193,      # 19.3% profit margin
        'debt_to_equity': 0.43,      # 0.43 debt-to-equity ratio
        'current_ratio': 1.8,        # 1.8 current ratio
        'return_on_equity': 0.124,   # 12.4% ROE
        'return_on_assets': 0.073    # 7.3% ROA
    }
    
    # Insert fundamentals data
    for item in sample_fundamentals:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fundamentals
                (id, ticker, date, total_revenue, gross_profit, operating_income, net_income, ebitda, 
                operating_cash_flow, free_cash_flow, total_assets, total_debt, total_equity, 
                current_assets, current_liabilities, working_capital)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                item['id'], item['ticker'], item['date'], item['total_revenue'], 
                item['gross_profit'], item['operating_income'], item['net_income'], 
                item['ebitda'], item['operating_cash_flow'], item['free_cash_flow'], 
                item['total_assets'], item['total_debt'], item['total_equity'], 
                item['current_assets'], item['current_liabilities'], item['working_capital']
            ])
        except Exception as e:
            logger.error(f"Error inserting fundamentals: {e}")
    
    # Insert fundamentals features
    try:
        conn.execute("""
            INSERT OR REPLACE INTO fundamentals_features
            (setup_id, revenue_growth_yoy, profit_margin, debt_to_equity, current_ratio, return_on_equity, return_on_assets)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            sample_features['setup_id'], sample_features['revenue_growth_yoy'], 
            sample_features['profit_margin'], sample_features['debt_to_equity'], 
            sample_features['current_ratio'], sample_features['return_on_equity'], 
            sample_features['return_on_assets']
        ])
    except Exception as e:
        logger.error(f"Error inserting fundamentals features: {e}")
    
    # Verify insertion
    count = conn.execute("SELECT COUNT(*) FROM fundamentals WHERE ticker = ?", [ticker]).fetchone()[0]
    logger.info(f"Added {count} fundamentals records for {ticker}")
    
    count = conn.execute("SELECT COUNT(*) FROM fundamentals_features WHERE setup_id = ?", [setup_id]).fetchone()[0]
    logger.info(f"Added {count} fundamentals features for {setup_id}")

def add_analyst_recommendations(conn, setup_id="AFN_2023-11-20", ticker="AFN"):
    """Add sample analyst recommendations"""
    logger.info(f"Adding sample analyst recommendations for {setup_id}")
    
    # Check if table exists
    table_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'analyst_recommendations'
    """).fetchone()[0]
    
    if not table_exists:
        logger.info("Creating analyst_recommendations table")
        conn.execute("""
            CREATE TABLE analyst_recommendations (
                id INTEGER PRIMARY KEY,
                setup_id VARCHAR,
                ticker VARCHAR,
                analyst_firm VARCHAR,
                analyst_name VARCHAR,
                recommendation VARCHAR,
                previous_recommendation VARCHAR,
                price_target DOUBLE,
                previous_price_target DOUBLE,
                report_date DATE,
                notes VARCHAR
            )
        """)
    
    # Sample analyst recommendations
    sample_data = [
        {
            'id': 4001,
            'setup_id': setup_id,
            'ticker': ticker,
            'analyst_firm': 'Morgan Stanley',
            'analyst_name': 'Sarah Johnson',
            'recommendation': 'Overweight',
            'previous_recommendation': 'Equal-weight',
            'price_target': 9.20,
            'previous_price_target': 8.50,
            'report_date': '2023-11-16',
            'notes': f"Upgraded {ticker} following strong Q3 results. Impressed by margin improvement and revenue growth acceleration."
        },
        {
            'id': 4002,
            'setup_id': setup_id,
            'ticker': ticker,
            'analyst_firm': 'Goldman Sachs',
            'analyst_name': 'Michael Chen',
            'recommendation': 'Buy',
            'previous_recommendation': 'Buy',
            'price_target': 9.50,
            'previous_price_target': 9.00,
            'report_date': '2023-11-16',
            'notes': f"Maintained Buy rating on {ticker} but raised price target. Highlighted robust cash generation and new contract win."
        },
        {
            'id': 4003,
            'setup_id': setup_id,
            'ticker': ticker,
            'analyst_firm': 'JP Morgan',
            'analyst_name': 'David Williams',
            'recommendation': 'Neutral',
            'previous_recommendation': 'Neutral',
            'price_target': 8.00,
            'previous_price_target': 7.80,
            'report_date': '2023-11-17',
            'notes': f"Maintained Neutral rating on {ticker}. Acknowledged improved performance but concerned about sector-wide challenges."
        }
    ]
    
    # Insert data
    for item in sample_data:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO analyst_recommendations
                (id, setup_id, ticker, analyst_firm, analyst_name, recommendation, previous_recommendation,
                price_target, previous_price_target, report_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                item['id'], item['setup_id'], item['ticker'], item['analyst_firm'], 
                item['analyst_name'], item['recommendation'], item['previous_recommendation'],
                item['price_target'], item['previous_price_target'], item['report_date'], 
                item['notes']
            ])
        except Exception as e:
            logger.error(f"Error inserting analyst recommendation: {e}")
    
    # Verify insertion
    count = conn.execute("SELECT COUNT(*) FROM analyst_recommendations WHERE setup_id = ?", [setup_id]).fetchone()[0]
    logger.info(f"Added {count} analyst recommendations for {setup_id}")

def add_user_posts(conn, setup_id="AFN_2023-11-20", ticker="AFN"):
    """Add sample user posts"""
    logger.info(f"Adding sample user posts for {setup_id}")
    
    # Check if table exists
    table_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'user_posts'
    """).fetchone()[0]
    
    if not table_exists:
        logger.info("Creating user_posts table")
        conn.execute("""
            CREATE TABLE user_posts (
                id INTEGER PRIMARY KEY,
                post_id VARCHAR,
                setup_id VARCHAR,
                ticker VARCHAR,
                user_handle VARCHAR,
                post_content VARCHAR,
                post_date TIMESTAMP,
                post_url VARCHAR,
                sentiment_score DOUBLE
            )
        """)
    
    # Sample user posts
    sample_data = [
        {
            'id': 5001,
            'post_id': f"{setup_id}-post-1",
            'setup_id': setup_id,
            'ticker': ticker,
            'user_handle': 'investor123',
            'post_content': f"Just saw {ticker}'s Q3 results - very impressive! Revenue up 15% and margins improving. This company continues to execute well in a challenging environment. I'm holding my position and might even add more at current levels.",
            'post_date': '2023-11-15 08:45:00',
            'post_url': f"https://example.com/forum/{ticker}/post/5001",
            'sentiment_score': 0.85
        },
        {
            'id': 5002,
            'post_id': f"{setup_id}-post-2",
            'setup_id': setup_id,
            'ticker': ticker,
            'user_handle': 'marketwatcher',
            'post_content': f"The new contract win for {ticker} looks significant. £12m over 3 years might not sound huge, but for a company this size it's material. Could open doors to more business in the tech sector too.",
            'post_date': '2023-11-17 10:30:00',
            'post_url': f"https://example.com/forum/{ticker}/post/5002",
            'sentiment_score': 0.75
        },
        {
            'id': 5003,
            'post_id': f"{setup_id}-post-3",
            'setup_id': setup_id,
            'ticker': ticker,
            'user_handle': 'skeptical_trader',
            'post_content': f"Not convinced about {ticker}'s recent rally. Yes, results were good but the valuation is getting stretched. The broader market looks shaky and this could pull back if we see sector rotation.",
            'post_date': '2023-11-18 15:20:00',
            'post_url': f"https://example.com/forum/{ticker}/post/5003",
            'sentiment_score': -0.3
        },
        {
            'id': 5004,
            'post_id': f"{setup_id}-post-4",
            'setup_id': setup_id,
            'ticker': ticker,
            'user_handle': 'value_seeker',
            'post_content': f"Interesting board changes at {ticker}. The new NED has a strong tech background which aligns well with their strategic direction. Good to see they're strengthening governance.",
            'post_date': '2023-11-18 16:45:00',
            'post_url': f"https://example.com/forum/{ticker}/post/5004",
            'sentiment_score': 0.4
        },
        {
            'id': 5005,
            'post_id': f"{setup_id}-post-5",
            'setup_id': setup_id,
            'ticker': ticker,
            'user_handle': 'chart_master',
            'post_content': f"$${ticker} breaking out of its trading range on high volume. The technical setup looks very bullish with MACD crossing over and RSI still not in overbought territory. Could see further upside from here.",
            'post_date': '2023-11-19 09:15:00',
            'post_url': f"https://example.com/forum/{ticker}/post/5005",
            'sentiment_score': 0.9
        }
    ]
    
    # Insert data
    for item in sample_data:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO user_posts
                (id, post_id, setup_id, ticker, user_handle, post_content, post_date, post_url, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                item['id'], item['post_id'], item['setup_id'], item['ticker'], 
                item['user_handle'], item['post_content'], item['post_date'], 
                item['post_url'], item['sentiment_score']
            ])
        except Exception as e:
            logger.error(f"Error inserting user post: {e}")
    
    # Verify insertion
    count = conn.execute("SELECT COUNT(*) FROM user_posts WHERE setup_id = ?", [setup_id]).fetchone()[0]
    logger.info(f"Added {count} user posts for {setup_id}")

def add_labels(conn, setup_id="AFN_2023-11-20"):
    """Add labels table if it doesn't exist (without adding actual labels for the test setup)"""
    logger.info("Ensuring labels table exists")
    
    # Check if table exists
    table_exists = conn.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_name = 'labels'
    """).fetchone()[0]
    
    if not table_exists:
        logger.info("Creating labels table")
        conn.execute("""
            CREATE TABLE labels (
                setup_id VARCHAR PRIMARY KEY,
                stock_return_10d DOUBLE,
                benchmark_return_10d DOUBLE,
                outperformance_10d DOUBLE,
                days_outperformed_10d INTEGER,
                benchmark_ticker VARCHAR,
                calculation_date TIMESTAMP,
                actual_days_calculated INTEGER
            )
        """)
        
        logger.info("Labels table created")
    else:
        logger.info("Labels table already exists")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Add sample data for testing')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                      help='Path to DuckDB database')
    parser.add_argument('--setup-id', default='AFN_2023-11-20',
                      help='Setup ID to add data for')
    parser.add_argument('--ticker', default='AFN',
                      help='Ticker symbol')
    
    args = parser.parse_args()
    
    logger.info(f"Adding sample data for {args.setup_id} ({args.ticker}) to {args.db_path}")
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Connect to DuckDB
    conn = duckdb.connect(args.db_path)
    
    try:
        # Add data
        add_rns_announcements(conn, args.setup_id, args.ticker)
        add_stock_news(conn, args.setup_id, args.ticker)
        add_fundamentals(conn, args.setup_id, args.ticker)
        add_analyst_recommendations(conn, args.setup_id, args.ticker)
        add_user_posts(conn, args.setup_id, args.ticker)
        add_labels(conn, args.setup_id)
        
        # Commit changes
        conn.commit()
        logger.info(f"Successfully added sample data for {args.setup_id}")
        
    except Exception as e:
        logger.error(f"Error adding sample data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 