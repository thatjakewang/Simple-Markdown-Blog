from flask import Flask, render_template, url_for, jsonify, send_from_directory, Response
from flask_flatpages import FlatPages
from datetime import datetime, date
import os

# ========== 載入應用程式 ==========
app = Flask(__name__)
# ========== FlatPages 配置 ==========
app.config['FLATPAGES_EXTENSION'] = '.md'
app.config['FLATPAGES_ROOT'] = 'posts'
app.config['FLATPAGES_MARKDOWN_EXTENSIONS'] = [
    'fenced_code',
    'codehilite',
]
pages = FlatPages(app)

CATEGORY_SLUGS = {
    '機器學習': 'machine-learning',
}

# ========== 自訂函式 ==========
# 日期格式轉換
@app.template_filter('strftime')
def format_datetime(value, format="%Y-%m-%d"):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.strftime(format)
    except (AttributeError, ValueError):
        return str(value)

# 日期轉字串
def format_date_str(raw_date):
    try:
        return raw_date.strftime('%Y-%m-%d') if raw_date else ''
    except (AttributeError, ValueError):
        return str(raw_date) if raw_date else ''

# 文章排序
def get_sorted_posts():
    """取得排序後的文章列表"""
    posts = [p for p in pages if 'date' in p.meta]
    try:
        posts.sort(key=lambda item: item['date'], reverse=True)
    except Exception as e:
        print(f"排序錯誤: {e}")
        pass
    return posts

# 名稱轉換
@app.template_filter('to_slug')
def category_to_slug(category_name):
    return CATEGORY_SLUGS.get(category_name, category_name)

# === ROUTES ===
# robots.txt
@app.route('/robots.txt')
def robots():
    return send_from_directory(app.static_folder, 'robots.txt')
# sitemap.xml
@app.route('/sitemap.xml')
def sitemap():
    """Generate sitemap.xml dynamically"""
    base_url = 'https://jake.tw'

    # Static pages
    sitemap_urls = [
        {'loc': base_url + '/', 'priority': '1.0'},
    ]

    # Blog posts
    for post in get_sorted_posts():
        sitemap_urls.append({
            'loc': base_url + url_for('post', path=post.path),
            'lastmod': format_date_str(post.meta.get('date')),
            'priority': '0.7'
        })
    
    # Build XML
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    
    for page in sitemap_urls:
        xml += '  <url>\n'
        xml += f'    <loc>{page["loc"]}</loc>\n'
        if page.get('lastmod'):
            xml += f'    <lastmod>{page["lastmod"]}</lastmod>\n'
        xml += f'    <priority>{page["priority"]}</priority>\n'
        xml += '  </url>\n'
    
    xml += '</urlset>'
    
    return Response(xml, mimetype='application/xml')

# HOMEPAGE
@app.route('/')
def home():
    recent_posts = get_sorted_posts()
    return render_template('index.html', posts=recent_posts)

# SINGLE POST PAGE
@app.route('/<path:path>/')
def post(path):
    page = pages.get_or_404(path)
    return render_template('post.html', page=page)

# Category
@app.route('/category/<slug>/')
def category(slug):
    category_name = next((zh for zh, en in CATEGORY_SLUGS.items() if en == slug), slug)

    category_posts = [
        p for p in get_sorted_posts() 
        if p.meta.get('category') == category_name
    ]

    return render_template('category.html', category_name=category_name, posts=category_posts)

#404
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404
#500
@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.run(debug=debug, port=5001, host='0.0.0.0')