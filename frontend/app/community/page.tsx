'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';

interface CommunityPost {
    id: string;
    text: string;
    author_display: string;
    author_uid?: string;
    created_at?: string;
    reactions_count?: { heart?: number; thumbs_up?: number };
}

interface CommunityPollOption {
    id: string;
    text: string;
    votes: number;
}

interface CommunityPoll {
    id: string;
    question: string;
    options: CommunityPollOption[];
    author_display: string;
    author_uid?: string;
    created_at?: string;
}

export default function CommunityPage() {
    const [posts, setPosts] = useState<CommunityPost[]>([]);
    const [polls, setPolls] = useState<CommunityPoll[]>([]);
    const [showPostModal, setShowPostModal] = useState(false);
    const [postText, setPostText] = useState('');
    const [showWelcome, setShowWelcome] = useState(false);

    const user = useAuthStore((state) => state.user);
    const authLoading = useAuthStore((state) => state.isLoading);
    const checkAuth = useAuthStore((state) => state.checkAuth);
    const router = useRouter();

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    useEffect(() => {
        if (authLoading) return;
        if (!user) {
            router.push('/login');
            return;
        }
        loadFeed();
        showWelcomeIfFirstVisit();
    }, [user, authLoading]);

    const showWelcomeIfFirstVisit = () => {
        try {
            const key = 'community_welcome_shown_v1';
            if (!localStorage.getItem(key)) {
                setShowWelcome(true);
            }
        } catch {
            setShowWelcome(false);
        }
    };

    const loadFeed = async () => {
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/community/feed`, {
                credentials: 'include',
            });
            const data = await response.json();
            if (!data || !data.ok) {
                setPosts([]);
                setPolls([]);
                return;
            }
            setPosts(data.posts || []);
            setPolls(data.polls || []);
        } catch (error) {
            console.error('Failed to load community feed:', error);
        }
    };

    const handleCreatePost = async () => {
        if (!postText.trim()) return alert('Please enter content');
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/community/post`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: postText }),
        });
        setPostText('');
        setShowPostModal(false);
        loadFeed();
    };

    const handleReact = async (postId: string, reaction: 'heart' | 'thumbs_up') => {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/community/post/react`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ post_id: postId, reaction }),
        });
        loadFeed();
    };

    const handleVote = async (pollId: string, optionId: string) => {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/community/poll/vote`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ poll_id: pollId, option_id: optionId }),
        });
        loadFeed();
    };

    const myPosts = posts.filter((post) => post.author_uid && post.author_uid === user?.id);
    const myPolls = polls.filter((poll) => poll.author_uid && poll.author_uid === user?.id);

    if (authLoading || !user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="loading-shimmer h-8 w-48 rounded-lg"></div>
            </div>
        );
    }

    return (
        <>
            <div className="max-w-6xl mx-auto space-y-6">
                <h2 className="text-3xl font-bold text-theme-text-main">Community</h2>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="space-y-4" id="posts-col">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xl font-semibold">Posts</h3>
                            <button id="open-post-modal" className="btn-primary" onClick={() => setShowPostModal(true)}>
                                New Post
                            </button>
                        </div>
                        <div id="posts" className="space-y-4">
                            {posts.length > 0 ? (
                                posts.map((post) => (
                                    <div key={post.id} className="community-post card">
                                        <div className="meta text-theme-text-subtle">
                                            <strong>{post.author_display}</strong> • {post.created_at || ''}
                                        </div>
                                        <div className="text mt-2 text-theme-text-main">{post.text}</div>
                                        <div className="actions mt-3 flex gap-2">
                                            <button className="react" onClick={() => handleReact(post.id, 'heart')}>
                                                ❤️ <span className="count">{post.reactions_count?.heart || 0}</span>
                                            </button>
                                            <button className="react" onClick={() => handleReact(post.id, 'thumbs_up')}>
                                                👍 <span className="count">{post.reactions_count?.thumbs_up || 0}</span>
                                            </button>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="muted text-theme-text-subtle">No posts yet — be the first to share.</div>
                            )}
                        </div>
                    </div>

                    <div className="space-y-4" id="polls-col">
                        <h3 className="text-xl font-semibold">Polls</h3>
                        <div id="polls" className="space-y-4">
                            {polls.length > 0 ? (
                                polls.map((poll) => (
                                    <div key={poll.id} className="community-poll card">
                                        <div className="meta text-theme-text-subtle">
                                            <strong>{poll.author_display}</strong> • {poll.created_at || ''}
                                        </div>
                                        <div className="question mt-2 text-theme-text-main">{poll.question}</div>
                                        <div className="options mt-2 space-y-2">
                                            {poll.options.map((opt) => (
                                                <div key={opt.id} className="opt">
                                                    <button className="vote" onClick={() => handleVote(poll.id, opt.id)}>
                                                        {opt.text} <span className="count">({opt.votes || 0})</span>
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="muted text-theme-text-subtle">No polls yet — create one to get suggestions.</div>
                            )}
                        </div>
                    </div>

                    <div className="space-y-4" id="my-col">
                        <h3 className="text-xl font-semibold">My Activity</h3>
                        <div id="my-posts" className="space-y-4">
                            <h4 className="text-theme-text-subtle">My Posts</h4>
                            {myPosts.map((post) => (
                                <div key={post.id} className="community-post card">
                                    <div className="meta text-theme-text-subtle">
                                        <strong>{post.author_display}</strong> • {post.created_at || ''}
                                    </div>
                                    <div className="text mt-2 text-theme-text-main">{post.text}</div>
                                </div>
                            ))}
                        </div>
                        <div id="my-polls" className="space-y-4">
                            <h4 className="text-theme-text-subtle">My Polls</h4>
                            {myPolls.map((poll) => (
                                <div key={poll.id} className="community-poll card">
                                    <div className="meta text-theme-text-subtle">
                                        <strong>{poll.author_display}</strong> • {poll.created_at || ''}
                                    </div>
                                    <div className="question mt-2 text-theme-text-main">{poll.question}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {showPostModal && (
                <div id="post-modal" className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
                    <div className="card max-w-2xl p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl">Create Post</h3>
                            <button id="post-modal-close" className="btn-primary" onClick={() => setShowPostModal(false)}>
                                Close
                            </button>
                        </div>
                        <div>
                            <textarea
                                id="modal-post-text"
                                rows={4}
                                className="w-full p-2 mb-2 bg-theme-panel text-theme-text-main"
                                placeholder="Write something..."
                                value={postText}
                                onChange={(event) => setPostText(event.target.value)}
                            ></textarea>
                            <div>
                                <button id="modal-post-submit" className="btn-primary" onClick={handleCreatePost}>
                                    Post
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {showWelcome && (
                <div id="community-welcome" className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
                    <div className="card max-w-2xl p-8 text-center">
                        <h2 className="text-2xl font-bold mb-4">Welcome to the Community</h2>
                        <p className="mb-6">This is a safe space to share how you feel and ask for suggestions. Choose one to get started.</p>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="card p-6 cursor-pointer" id="welcome-post" onClick={() => setShowPostModal(true)}>
                                <h3 className="font-semibold">Posts</h3>
                                <p className="text-sm mt-2">Share your feelings, short updates or thoughts — react or respond kindly.</p>
                            </div>
                            <div className="card p-6 cursor-pointer" id="welcome-poll" onClick={() => setShowWelcome(false)}>
                                <h3 className="font-semibold">Polls</h3>
                                <p className="text-sm mt-2">Ask a question and offer options for others to vote on and suggest ideas.</p>
                            </div>
                        </div>
                        <div className="mt-6">
                            <button
                                id="welcome-close"
                                className="btn-primary"
                                onClick={() => {
                                    localStorage.setItem('community_welcome_shown_v1', '1');
                                    setShowWelcome(false);
                                }}
                            >
                                Maybe later
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
