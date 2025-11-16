from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from google.cloud import firestore
from google.api_core import exceptions as g_exceptions
import traceback
from translations.translation_utils import get_user_translations

community_bp = Blueprint('community', __name__)


def _date_and_anon_suffix(ts):
    """Return (date_str, anon_display) where date_str is YYYY-MM-DD and
    anon_display is 'Anonymous <num>' generated from timestamp seconds.
    """
    date_str = ''
    try:
        if hasattr(ts, 'date'):
            # datetime-like
            date_str = ts.date().isoformat()
        elif hasattr(ts, 'isoformat'):
            date_str = ts.isoformat().split('T')[0]
        else:
            s = str(ts)
            date_str = s.split('T')[0] if 'T' in s else s.split(' ')[0]
    except Exception:
        date_str = ''

    # create a short alphanumeric id from timestamp seconds (base36, 4 chars)
    def _to_base36(n):
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if n == 0:
            return '0'
        out = ''
        while n > 0:
            n, r = divmod(n, 36)
            out = chars[r] + out
        return out

    suffix = '0000'
    try:
        if hasattr(ts, 'timestamp'):
            sec = int(ts.timestamp())
        else:
            sec = abs(hash(str(ts)))
        val = sec % (36 ** 4)  # fit into 4 base36 chars
        s = _to_base36(val).rjust(4, '0')
        suffix = s
    except Exception:
        suffix = '0000'

    return date_str, f'Anonymous {suffix}'


@community_bp.route('/community', methods=['GET'])
def community_page():
    # Show the welcome/choice page; user then navigates to posts or polls pages
    from flask import render_template
    return render_template('community_welcome.html', lang=get_user_translations(), user=current_user)


@community_bp.route('/community/posts', methods=['GET'])
def community_posts_page():
    # Render posts-only page with initial posts
    from flask import render_template
    initial_posts = []
    try:
        from app import db
        posts_ref = db.collection('community_posts').order_by('created_at', direction=firestore.Query.DESCENDING).limit(50)
        for doc in posts_ref.stream():
            d = doc.to_dict() or {}
            ts = d.get('created_at')
            created, anon = _date_and_anon_suffix(ts)
            stored = d.get('author_display')
            display = stored if stored and stored != 'Anonymous' else anon
            initial_posts.append({
                'id': doc.id,
                'text': d.get('text', ''),
                'author_display': display,
                'author_uid': d.get('author_uid'),
                'created_at': created,
                'reactions_count': d.get('reactions_count', {'heart': 0, 'thumbs_up': 0})
            })
    except Exception:
        initial_posts = []
    return render_template('community_posts.html', lang=get_user_translations(), user=current_user, posts=initial_posts)


@community_bp.route('/community/polls', methods=['GET'])
def community_polls_page():
    from flask import render_template
    initial_polls = []
    try:
        from app import db
        polls_ref = db.collection('community_polls').order_by('created_at', direction=firestore.Query.DESCENDING).limit(50)
        for doc in polls_ref.stream():
            d = doc.to_dict() or {}
            ts = d.get('created_at')
            created, anon = _date_and_anon_suffix(ts)
            # convert options dict -> list of {id, text, votes} for templates
            opts = []
            raw_opts = d.get('options', {}) or {}
            try:
                # keep stable order by sorting keys like opt0, opt1
                for k in sorted(raw_opts.keys()):
                    v = raw_opts.get(k, {}) or {}
                    opts.append({'id': k, 'text': v.get('text', ''), 'votes': v.get('votes', 0)})
            except Exception:
                opts = []

            stored = d.get('author_display')
            display = stored if stored and stored != 'Anonymous' else anon
            initial_polls.append({
                'id': doc.id,
                'question': d.get('question', ''),
                'options': opts,
                'author_display': display,
                'author_uid': d.get('author_uid'),
                'created_at': created
            })
    except Exception:
        initial_polls = []
    return render_template('community_polls.html', lang=get_user_translations(), user=current_user, polls=initial_polls)


@community_bp.route('/api/community/feed', methods=['GET'])
def community_feed():
    # API endpoints should return JSON errors for unauthenticated requests
    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return jsonify({'ok': False, 'error': 'unauthenticated'}), 401
    try:
        from app import db
        posts = []
        polls = []

        # fetch recent posts
        posts_ref = db.collection('community_posts').order_by('created_at', direction=firestore.Query.DESCENDING).limit(30)
        for doc in posts_ref.stream():
            d = doc.to_dict() or {}
            ts = d.get('created_at')
            created, anon = _date_and_anon_suffix(ts)
            stored = d.get('author_display')
            display = stored if stored and stored != 'Anonymous' else anon
            posts.append({
                'id': doc.id,
                'text': d.get('text', ''),
                'author_display': display,
                'author_uid': d.get('author_uid'),
                'created_at': created,
                'reactions_count': d.get('reactions_count', {'heart': 0, 'thumbs_up': 0})
            })

        polls_ref = db.collection('community_polls').order_by('created_at', direction=firestore.Query.DESCENDING).limit(20)
        for doc in polls_ref.stream():
            d = doc.to_dict() or {}
            ts = d.get('created_at')
            created, anon = _date_and_anon_suffix(ts)
            # convert to list for API consumer
            opts = []
            raw_opts = d.get('options', {}) or {}
            try:
                for k in sorted(raw_opts.keys()):
                    v = raw_opts.get(k, {}) or {}
                    opts.append({'id': k, 'text': v.get('text', ''), 'votes': v.get('votes', 0)})
            except Exception:
                opts = []

            stored = d.get('author_display')
            display = stored if stored and stored != 'Anonymous' else anon
            polls.append({
                'id': doc.id,
                'question': d.get('question', ''),
                'options': opts,
                'author_display': display,
                'author_uid': d.get('author_uid'),
                'created_at': created
            })

        # log counts for debugging
        try:
            import logging
            logging.getLogger('community').info(f'community_feed: posts={len(posts)} polls={len(polls)} for user={current_user.id}')
        except Exception:
            pass
        return jsonify({'ok': True, 'posts': posts, 'polls': polls})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@community_bp.route('/api/community/post', methods=['POST'])
def create_post():
    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return jsonify({'ok': False, 'error': 'unauthenticated'}), 401
    data = request.get_json() or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': 'Empty post'}), 400
    if len(text) > 10000:
        return jsonify({'ok': False, 'error': 'Post too long'}), 413
    try:
        from app import db
        post = {
            'author_uid': current_user.id,
            'author_display': 'Anonymous',
            'text': text,
            'created_at': firestore.SERVER_TIMESTAMP,
            'reactions_count': {'heart': 0, 'thumbs_up': 0}
        }
        doc_ref = db.collection('community_posts').document()
        doc_ref.set(post)
        post_id = doc_ref.id
        return jsonify({'ok': True, 'post_id': post_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@community_bp.route('/api/community/post/react', methods=['POST'])
def react_post():
    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return jsonify({'ok': False, 'error': 'unauthenticated'}), 401
    data = request.get_json() or {}
    post_id = data.get('post_id')
    reaction = data.get('reaction')
    if reaction not in ('heart', 'thumbs_up'):
        return jsonify({'ok': False, 'error': 'Invalid reaction'}), 400
    if not post_id:
        return jsonify({'ok': False, 'error': 'post_id required'}), 400
    try:
        from app import db

        post_ref = db.collection('community_posts').document(post_id)
        reactions_ref = post_ref.collection('reactions').document(current_user.id)

        # Non-transactional optimistic update
        post_snap = post_ref.get()
        if not post_snap.exists:
            return jsonify({'ok': False, 'error': 'Post not found'}), 404
        post_data = post_snap.to_dict() or {}
        counts = post_data.get('reactions_count', {'heart': 0, 'thumbs_up': 0})

        react_snap = reactions_ref.get()
        if react_snap.exists:
            prev = react_snap.to_dict().get('reaction')
            if prev == reaction:
                # toggle off
                reactions_ref.delete()
                counts[reaction] = max(0, counts.get(reaction, 0) - 1)
            else:
                # change reaction
                reactions_ref.set({'reaction': reaction, 'created_at': firestore.SERVER_TIMESTAMP})
                counts[prev] = max(0, counts.get(prev, 0) - 1)
                counts[reaction] = counts.get(reaction, 0) + 1
        else:
            # add reaction
            reactions_ref.set({'reaction': reaction, 'created_at': firestore.SERVER_TIMESTAMP})
            counts[reaction] = counts.get(reaction, 0) + 1

        post_ref.update({'reactions_count': counts})
        return jsonify({'ok': True, 'reactions_count': counts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@community_bp.route('/api/community/poll', methods=['POST'])
def create_poll():
    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return jsonify({'ok': False, 'error': 'unauthenticated'}), 401
    data = request.get_json() or {}
    question = (data.get('question') or '').strip()
    options = data.get('options') or []
    if not question or not options or not isinstance(options, list) or len(options) < 2:
        return jsonify({'ok': False, 'error': 'Question and at least 2 options required'}), 400
    try:
        from app import db
        opts_map = {}
        for i, opt in enumerate(options):
            opts_map[f'opt{i}'] = {'text': opt, 'votes': 0}
        poll = {
            'author_uid': current_user.id,
            'author_display': 'Anonymous',
            'question': question,
            'options': opts_map,
            'created_at': firestore.SERVER_TIMESTAMP
        }
        doc_ref = db.collection('community_polls').document()
        doc_ref.set(poll)
        poll_id = doc_ref.id
        return jsonify({'ok': True, 'poll_id': poll_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@community_bp.route('/api/community/poll/vote', methods=['POST'])
def vote_poll():
    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return jsonify({'ok': False, 'error': 'unauthenticated'}), 401
    data = request.get_json() or {}
    poll_id = data.get('poll_id')
    option_id = data.get('option_id')
    if not poll_id or not option_id:
        return jsonify({'ok': False, 'error': 'poll_id and option_id required'}), 400
    try:
        from app import db
        poll_ref = db.collection('community_polls').document(poll_id)
        votes_ref = poll_ref.collection('votes').document(current_user.id)

        # validate poll and option
        poll_snap = poll_ref.get()
        if not poll_snap.exists:
            return jsonify({'ok': False, 'error': 'Poll not found'}), 404
        poll_data = poll_snap.to_dict() or {}
        options = poll_data.get('options', {})
        if option_id not in options:
            return jsonify({'ok': False, 'error': 'Invalid option'}), 400

        # attempt to create vote doc; if exists, user already voted
        try:
            votes_ref.create({'option_id': option_id, 'created_at': firestore.SERVER_TIMESTAMP})
        except Exception as e:
            if isinstance(e, g_exceptions.AlreadyExists) or 'AlreadyExists' in str(e):
                return jsonify({'ok': False, 'error': 'User already voted'}), 400
            raise

        # increment nested option vote counter
        field_path = f'options.{option_id}.votes'
        poll_ref.update({field_path: firestore.Increment(1)})

        poll_snap = poll_ref.get()
        return jsonify({'ok': True, 'options': poll_snap.to_dict().get('options', {})})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500
