#include <iostream>
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

vvl warshall_floyd(vvl E) {
    ll H = E.size(), W = E[0].size();
    assert(H == W);
    rep(h, H) assert(E[h][h] == 0);
    vvl retval = E;
    rep(i, H) {
        rep(h, H) {
            rep(w, H) {
                chmin(retval[h][w], retval[h][i] + retval[i][w]);
            }
        }
    }
    return retval;
}

class Eratosthenes {
    // 注意: エラトステネスの篩は，大きい数（10^6以上）に対しては使えない
    // 大きい数を素因数分解するときは，↓のprime_factorization関数を使う
    private:
        ll max_val;
        vl table;
    public:
        Eratosthenes(ll max_val = 1000000) : max_val(max_val), table(max_val+1, inf) {
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
        }
        vl get_table() { return table; }
        vl prime_factorization(ll n) {
            // nを素因数分解し，素因数を列挙する
            // o(log(n))
            assert(n <= max_val);
            vl ans;
            while (n != 1) {
                ans.push_back(table[n]);
                n /= table[n];
            }
            return ans;
        }
        vl get_primes(ll n) {
            // n以下の素数を列挙する
            // o(n)
            assert(n <= max_val);
            vl ans;
            FOR(i, 2, n+1) {
                if (table[i] == i) ans.push_back(i);
            }
            return ans;
        }
        bool is_prime(ll n) {
            // nが素数かどうか判定する
            // o(1)
            assert(n <= max_val);
            return table[n] == n;
        }
        ll get_max_val() { return max_val; }
};

vl prime_factorization(ll n) {
    // nを素因数分解し，素因数を列挙する
    // o(sqrt(n))
    ll rem = n;
    vl ans;
    for (ll i = 2; i*i <= n; i++) {
        while (rem % i == 0) {
            ans.push_back(i);
            rem /= i;
        }
    }
    if (rem != 1) ans.push_back(rem);
    return ans;
}

vvvl kruskal(vvvl &edges) {
    ll N = edges.size();
    vvl edges_list;
    rep(n, N) {
        for (vl e : edges[n]) {
            ll cost = e[0];
            ll to = e[1];
            edges_list.push_back({cost, n, to});
        }
    }
    sort(all(edges_list));
    dsu d(N);
    vvvl ans(N);
    for (vl e : edges_list) {
        ll cost = e[0], from = e[1], to = e[2];
        if (d.same(from, to)) continue;
        d.merge(from, to);
        ans[from].push_back({cost, to});
    }
    if (d.groups().size() > 1) {
        return {};
    }
    return ans;

}

class Compression {
public:
    vl A;
    ml B;
    Compression(){}
    void add(ll a) {
        A.push_back(a);
    }
    void build() {
        sort(all(A));
        A.erase(unique(all(A)), A.end());
        rep(i, A.size()) {
            B[A[i]] = i;
        }
    }
    ll idx(ll a) {
        if (B.find(a) == B.end()) {
            cout << "error" << endl;
            exit(1);
        }
        return B[a];
    }

    ll val(ll idx) {
        return A[idx];
    }

    ll size() {
        return B.size();
    }
};

class GridCumulativeSum {
    private:
        vvl original;
        vvl cum_table;
        ll H, W;
    public:
    GridCumulativeSum(vvl original) : original(original) {
        H = original.size();
        W = original[0].size();
        cum_table = vvl(H+1, vl(W+1, 0));
        rep(h, H) {
            rep(w, W) {
                cum_table[h+1][w+1] = cum_table[h+1][w] + cum_table[h][w+1] - cum_table[h][w] + original[h][w];
            }
        }
    }
    ll query(ll h, ll dh, ll w, ll dw) {
        // [h, h+dh) x [w, w+dw)の範囲の和を求める
        assert(h+dh <= H && w+dw <= W);
        assert(dh > 0 && dw > 0);
        assert(h >= 0 && w >= 0);
        return cum_table[h+dh][w+dw] - cum_table[h][w+dw] - cum_table[h+dh][w] + cum_table[h][w];
    }
};

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

struct Node {
    ll val;
    Node *parent;
    vector<Node*> children;
};

// ll op(ll a, ll b){ return std::min(a, b); }
// ll e(){ return int(1e9)+1; }
// ll mapping(ll f, ll x){ return x+f; }
// ll composition(ll f, ll g){ return f+g; }
// ll id(){ return 0; }
// lazy_segtree<ll, op, e, ll, mapping, composition, id> seg(N);
// https://atcoder.github.io/ac-library/document_ja/lazysegtree.html
// 区間最小・区間和

// 和を計算するseg木
// ll op(ll a, ll b){ return a + b; }
// ll e(){ return 0; }
// segtree<ll, op, e> seg(N);

// 区間加算操作・区間和取得のlazy_segtree
// struct S{
//     ll value;
//     ll size;
// };
// using F = ll;
// S op(S a, S b){ return {a.value+b.value, a.size+b.size}; }
// S e(){ return {0, 0}; }
// S mapping(F f, S x){ return {x.value+x.size*f, x.size}; }
// F composition(F f, F g){ return f+g; }
// F id(){ return 0; }
//使用例
// int main(){
//     ll n = 5;
//     vector<S> v(n, {0, 1});
//     lazy_segtree<S, op, e, F, mapping, composition, id> seg(v);

//     seg.apply(l, r, v)//A[l]からA[r-1]までの全ての要素にvを足す
// }
 
// 転倒数
// ll op(ll a, ll b){ return a + b; }
// ll e(){ return 0; }
// ll N;cin>>N;
// vl A = VL(N);
// ll ans = 0;
// segtree<ll, op, e> seg(N);
// rep(i, N) {
//     ans += i - seg.prod(0, A[i]+1);
//     seg.set(A[i], seg.get(A[i]) + 1);
// }
// print(ans);

/* LCA(G, root): 木 G に対する根を root として Lowest Common Ancestor を求める構造体
    query(u,v): u と v の LCA を求める。計算量 O(logn)
    前処理: O(nlogn)時間, O(nlogn)空間
*/
struct LCA {
    vector<vector<ll>> parent;  // parent[k][u]:= u の 2^k 先の親
    vector<ll> dist;            // root からの距離
    LCA(const vvl &G, ll root = 0) { init(G, root); } // 有向グラフ(木構造)で初期化する
    // 初期化
    void init(const vvl &G, ll root = 0) {
        ll V = G.size();
        ll K = 1;
        while ((1 << K) < V) K++;
        parent.assign(K, vector<ll>(V, -1));
        dist.assign(V, -1);
        dfs(G, root, -1, 0);
        for (ll k = 0; k + 1 < K; k++) {
            for (ll v = 0; v < V; v++) {
                if (parent[k][v] < 0) {
                    parent[k + 1][v] = -1;
                } else {
                    parent[k + 1][v] = parent[k][parent[k][v]];
                }
            }
        }
    }
    // 根からの距離と1つ先の頂点を求める
    void dfs(const vvl &G, ll v, ll p, ll d) {
        parent[0][v] = p;
        dist[v] = d;
        for (auto e : G[v]) {
            if (e != p) dfs(G, e, v, d + 1);
        }
    }
    ll query(ll u, ll v) {
        if (dist[u] < dist[v]) swap(u, v);  // u の方が深いとする
        ll K = parent.size();
        // LCA までの距離を同じにする
        for (ll k = 0; k < K; k++) {
            if ((dist[u] - dist[v]) >> k & 1) {
                u = parent[k][u];
            }
        }
        // 二分探索で LCA を求める
        if (u == v) return u;
        for (ll k = K - 1; k >= 0; k--) {
            if (parent[k][u] != parent[k][v]) {
                u = parent[k][u];
                v = parent[k][v];
            }
        }
        return parent[0][u];
    }
};


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
{std::cin.tie(0)->sync_with_stdio(0);

    


    return 0;

}