#include <vector>
#include <iostream>

#include <torch/extension.h>
#include <limits>

namespace py = pybind11;
typedef std::vector<std::pair<int, int>> index_vec;


/* Store q, argsort(q), eta and accessors to them. */
class PruningQPData {
public:
    PruningQPData(torch::Tensor q_, const float eta_)
        : n_nodes{q_.size(0)}
        , n_samples{q_.size(1)}
        , eta_{eta_}
        , q_{q_}
        , q_srt_{q_.argsort(1, true)} // heavy lifting!
        , q{q_.accessor<float, 2>()}
        , q_srt{q_srt_.accessor<long, 2>()}
    { }

    const int64_t n_nodes;
    const int64_t n_samples;
    const float eta_;
    torch::Tensor q_;
    torch::Tensor q_srt_;
    torch::TensorAccessor<float, 2> q;
    torch::TensorAccessor<long, 2> q_srt;
};

/* Solution state, for computing backward pass */
class PruningQPState {
public:
    PruningQPState(const size_t n_nodes)
        : color{std::vector<long>(n_nodes)}
        , color_to_ix{std::vector<index_vec>(n_nodes)}
        , denoms{std::vector<int>(n_nodes)}
    {
        std::iota(color.begin(), color.end(), 0);
    };

    std::vector<long> color;
    std::vector<index_vec> color_to_ix;
    std::vector<int> denoms;
};


std::tuple<float, int, index_vec>
subproblem_init(const PruningQPData& data, const int j) {
    const auto& q_j = data.q[j];
    const auto& q_srt_j = data.q_srt[j];

    float dd = 0;
    float dd_init = dd;
    float denom = data.eta_;

    auto selected_indices = index_vec();

    for (int k = 0; k < data.n_samples; ++k)
    {
        const auto kk = q_srt_j[k];
        const auto qq = q_j[kk];

        if (qq < dd_init - 0.5) // changed new quadratic relaxation
            break;
        if (dd > denom * (qq + 0.5)) // changed new quadratic relaxation
            break;

        dd += qq + 0.5; // changed new quadratic relaxation
        denom += 1;
        selected_indices.push_back(std::make_pair(j, kk));
    }

    return {dd / denom, denom, selected_indices};
}

std::tuple<float, int, index_vec>
subproblem(
        const PruningQPData& data,
        const std::vector<long>& color,
        const int c
){
    const auto n_nodes = data.n_nodes;
    const auto n_samples = data.n_samples;

    // initialize dd_init and denom over
    // all nodes matching the color
    float dd_init = 0;
    int n_rows = 0;

    auto js = std::vector<int>();
    auto ks = std::vector<int>();

    for (int j = 0; j < n_nodes; ++j) {
        if (color[j] == c) {
            js.push_back(j);
            ks.push_back(0);
            n_rows += 1;
        }
    }

    //assert(n_rows >= 1);

    auto dd = dd_init;
    float denom = n_rows * data.eta_;

    auto selected_indices = index_vec();

    while (true) {

        // get largest q among all selected rows
        auto qmax = -std::numeric_limits<float>::infinity();
        auto jjmax = -1;
        auto jmax = -1;
        auto kkmax = -1;
        for (int jj = 0; jj < n_rows; ++jj) {
            auto j = js[jj];
            auto k = ks[jj];

            if (k >= n_samples)
                continue;

            auto kk = data.q_srt[j][k];
            auto qq = data.q[j][kk];

            if (qq > qmax) {
                qmax = qq;
                jjmax = jj;
                jmax = j;
                kkmax = kk;
            }
        }

        if (jjmax < 0)
            break;
        if (dd_init > n_rows * (qmax + 0.5)) // // changed new quadratic relaxation
            break;
        if (dd > denom * (qmax + 0.5)) // changed new quadratic relaxation
            break;

        dd += qmax + 0.5; // changed new quadratic relaxation
        denom += 1;
        ks[jjmax] += 1; // advance jth row head
        selected_indices.push_back(std::make_pair(jmax, kkmax));
    }
    return {dd / denom, denom, selected_indices};
}


std::tuple<torch::Tensor, PruningQPState>
compute_d_fast(torch::Tensor q, const float eta) {

    q = q.t();
    auto data = PruningQPData{q, eta};

    auto parent = [](int t) {return (t - 1) / 2;} ;

    // initialize solution
    auto sol = PruningQPState(data.n_nodes);

    // output
    torch::Tensor d = torch::zeros(data.n_nodes);
    auto d_acc = d.accessor<float, 1>();

    // solve all nested qps to initialize
    for (int j = 0; j < data.n_nodes; ++j) {
        std::tie(d_acc[j],
                 sol.denoms[j],
                 sol.color_to_ix[j]) = subproblem_init(data, j);
    }

    int n_iter = 0;

    while (true) {
        n_iter += 1;

        // find most violating edge
        int max_viol_ix = -1;
        float max_viol_d = -std::numeric_limits<float>::infinity();
        for (int t = 1; t < data.n_nodes; ++t) {
            int p = parent(t);
            if (d_acc[t] > d_acc[p] && d_acc[t] > max_viol_d) {
                max_viol_d = d_acc[t];
                max_viol_ix = t;
            }
        }

        if (max_viol_ix < 0) // no violation
            break;

        int t = max_viol_ix;
        int p = parent(t);
        int pc = sol.color[p];
        int tc = sol.color[t];

        // could refactor into sol.discard_color(tc);
        sol.color_to_ix[tc] = {};
        sol.denoms[tc] = 0;

        // could refactor into sol.merge(tc, pc);
        for (auto& c : sol.color) {
            if (c == tc)
                c = pc;
        }

        // update d[color == pc];
        float dcol;
        std::tie(dcol,
                 sol.denoms[pc],
                 sol.color_to_ix[pc]) = subproblem(data, sol.color, pc);

        for(int j = 0; j < data.n_nodes; ++j) {
            if (sol.color[j] == pc)
                d_acc[j] = dcol;
        }
    }
    return {d, sol};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_d_fast", &compute_d_fast, "todo");
    py::class_<PruningQPState>(m, "PruningQPState")
        .def_readonly("color", &PruningQPState::color)
        .def_readonly("denoms", &PruningQPState::denoms)
        .def_readonly("color_to_ix", &PruningQPState::color_to_ix);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
