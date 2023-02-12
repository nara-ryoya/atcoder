#include <atcoder/all>
#include <bits/stdc++.h>
using namespace std;
using namespace atcoder;
using ll = long long;
using vl = vector<ll>;
using vvl = vector<vl>;
using vvvl = vector<vvl>;
using ld = long double;
using pl = pair<ll, ll>;
using ml = map<ll, ll>;
using sl = set<ll>;
using vb = vector<bool>;
using vvb = vector<vector<bool>>;
using Grid = vector<string>;
using vs = vector<string>;
using heapl = priority_queue<ll, vl, greater<ll>>;
constexpr ll mod = 1000000007;
constexpr ll mod2 = 998244353;
constexpr ll inf = numeric_limits<ll>::max();
#define umap unordered_map
#define rep(i, n) for (ll i = 0; i < (n); i++)
#define FOR(i, n, m) for (ll i = n; i < (m); ++i)
#define all(A) A.begin(), A.end()
#define MP make_pair

vl VL(ll N)
{
    vl A(N);
    rep(i, N) cin >> A[i];
    return A;
}
Grid GI(ll H, ll W)
{
    Grid G(H);
    string s;
    rep(i, H)
    {
        cin >> s;
        G[i] = s;
    }
    return G;
}
template <typename P>
void print(P p) { cout << p << endl; }
template <typename P>
void print_v(const vector<P> &v)
{
    for (auto iter = v.begin(); iter != v.end(); ++iter)
    {
        cout << *iter << " ";
    }
    cout << endl;
}
template <typename P>
void print_vv(const vector<vector<P>> &vv)
{
    for (auto iteriter = vv.begin(); iteriter != vv.end(); ++iteriter)
    {
        auto v = *iteriter;
        for (auto iter = v.begin(); iter != v.end(); ++iter)
        {
            cout << *iter << " ";
        }
        cout << endl;
    }
}
template <typename P>
void print_vvv(const vector<vector<P>> &vvv)
{
    for (auto iteriteriter = vvv.begin(); iteriteriter != vvv.end(); ++iteriteriter)
    {
        auto vv = *iteriteriter;
        cout << '{';
        for (auto iteriter = vv.begin(); iteriter != vv.end(); ++iteriter)
        {
            auto v = *iteriter;
            cout << '{';
            for (auto iter = v.begin(); iter != v.end(); ++iter)
            {
                cout << *iter << ", ";
            }
            cout << "} ";
        }
        cout << '}';
        cout << endl;
    }
}
template <typename P>
P sum(const vector<P> &v)
{
    P sum = 0;
    for (P a : v)
    {
        sum += a;
    }
    return sum;
}
template <typename P, typename Q>
void print_map(const map<P, Q> mp)
{
    for (auto it = mp.begin(); it != mp.end(); ++it)
    {
        cout << '(' << (*it).first << ", " << (*it).second << ')' << ", ";
    }
    cout << endl;
}
template <typename P>
void print_set(const set<P> s)
{
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        cout << *it << " ";
    }
    cout << endl;
}
template <typename P>
void print_mset(const multiset<P> s)
{
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        cout << *it << " ";
    }
    cout << endl;
}
template <typename P, typename Q>
P Mod(P n, Q M)
{
    n %= M;
    if (n < 0)
        n += M;
    return n;
}
int ctoi(char c) { return (int)(c - '0'); }
int itoc(int i) { return (char)i + '0'; }
template <typename T>
bool chmax(T &a, const T &b)
{
    if (a < b)
    {
        a = b;
        return true;
    }
    return false;
}
template <typename T>
bool chmin(T &a, const T &b)
{
    if (a > b)
    {
        a = b;
        return true;
    }
    return false;
}
template <typename P>
P find_minimum_diff(vector<P> &vec, P target)
{
    auto it = lower_bound(all(vec), target);
    if (it == vec.begin())
        return *it - target;
    if (it == vec.end())
        return target - *(it - 1);
    return min(target - *(it - 1), (*it) - target);
}


class tsort {
public:
	ll V;
	vvl G;
	vl deg,res;
	tsort(ll node_size) : V(node_size), G(V), deg(V, 0){}
	void add_edge(ll from,ll to){
		G[from].push_back(to);
		deg[to]++;
	}
	bool solve() {
		queue<ll> que;
		for(ll i = 0; i < V; i++){
			if(deg[i] == 0){
				que.push(i);
			}
		}
		while(!que.empty()){
			ll p = que.front();
			que.pop();
			res.push_back(p);
			for(ll v : G[p]){
				if(--deg[v] == 0){
					que.push(v);
				}
			}
		}
		return (*max_element(deg.begin(),deg.end()) == 0);
	}
};

vl bfs(vvl &edges, ll N, ll start){
    queue<ll> q;
    vl dist(N, inf);
    dist[start] = 0;
    q.push(start);
    while (!q.empty()){
        ll p = q.front();
        q.pop();
        for (auto e : edges[p]){
            if (dist[e] == inf){
                dist[e] = 1 + dist[p];
                q.push(e);
            }
        }
    }
    return dist;
}

vvl bfs_grid(Grid &g, ll H, ll W, ll sy, ll sx, char wall='#'){
    queue<vl> que;
    que.push({sy, sx});
    vvl dyxs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    vvl dist(H, vl(W, inf));
    dist[sy][sx] = 0;
    while (!que.empty()){
        vl p = que.front();
        que.pop();
        ll dist_p = dist[p[0]][p[1]];
        for (vl dyx : dyxs){
            ll new_y = p[0]+dyx[0], new_x = p[1]+dyx[1];
            if (new_y<0 || new_y>=H || new_x<0 || new_x>=W) continue;
            if (g[new_y][new_x] != '#' && dist[new_y][new_x] == inf){
                dist[new_y][new_x] = dist_p+1;
                que.push({new_y, new_x});
            }
        }
    }
    return dist;
}

vl dijkstra(vvvl &edges, ll N, ll start)
{
    vl dist(N); rep(i, N) dist[i] = inf;

    priority_queue<vl, vvl, greater<vl>> P;
    P.push({0, start});
    bool seen[N]; rep(i, N) seen[i] =false;
    dist[start] = 0;
    while (!P.empty()){
        vl p = P.top();
        P.pop();
        ll cur_node = p[1];
        if (seen[cur_node]) continue;
        for (auto e : edges[cur_node]){
            if (!seen[e[0]]){
                if (dist[e[0]] > dist[cur_node] + e[1]){
                    dist[e[0]] = dist[cur_node] + e[1];
                    P.push({dist[e[0]], e[0]});
                }
            }
        }
        seen[cur_node] = true;
    }
    return dist;
}

vl minimum_prime_table(ll max_val) {
    vl table(max_val+1, inf);
    table[0] = -1;
    table[1] = -1;
    FOR(n, 2, max_val+1) {
        if (table[n] == inf) {
            table[n] = n;
            ll nn = n * 2;
            while (nn < max_val+1) {
                chmin(table[nn], n);
                nn+=n;
            }
        }
    }
    return table;
}

vl LIS(ll N, vl &A) {
    assert(A.size() == N);
    vl dp(N+2, inf);
    dp[0] = -inf;
    for (ll a : A) {
        auto iter = lower_bound(all(dp), a);
        dp[distance(dp.begin(), iter)] = a;
    }
    // ll N;cin>>N;
    // vl A = VL(N);
    // vl dp = LIS(N, A);
    // // print_v(dp);
    // rep(i, N+2) {
    //     if (dp[i] == inf) {
        // return i-1:
    //     }
    // }
    return dp;
}

// ll op(ll a, ll b){ return std::min(a, b); }
// ll e(){ return int(1e9)+1; }
// ll mapping(ll f, ll x){ return x+f; }
// ll composition(ll f, ll g){ return f+g; }
// ll id(){ return 0; }
// lazy_segtree<ll, op, e, ll, mapping, composition, id> seg(N);
// https://atcoder.github.io/ac-library/document_ja/lazysegtree.html
// 区間最小・区間和

bool dbg = false;
#define _dbg if (dbg)

struct Comp
{

    template <typename P>
    bool operator()(vector<P> p1, vector<P> p2) {
        if (p1[0] != p2[0]) return p1[0] > p2[0];
        if (p1[2] != p2[2]) return p1[2] < p2[2];
        return p1[1] <= p2[1];
    };
};

int main()
{
    



    return 0;

}