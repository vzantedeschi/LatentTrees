#include <vector>
#include <iostream>

#include <torch/extension.h>
#include <limits>

namespace py = pybind11;
typedef std::vector<std::pair<int, int>> index_vec;


std::tuple<float, int, index_vec> subproblem_init(
    torch::TensorAccessor<float, 1> eta,
    torch::TensorAccessor<float, 2> q,
    torch::TensorAccessor<long, 2> q_srt,
    const int j
){
    const auto n_samples = q.size(1);
    auto q_j = q[j];
    auto q_srt_j = q_srt[j];

    float dd = eta[j];
    float dd_init = dd;
    int denom = 1;

    auto selected_indices = index_vec();

    for (int k = 0; k < n_samples; ++k)
    {
        auto kk = q_srt_j[k];
        auto qq = q_j[kk];

        if (qq < dd_init) // is this check redundant?
            break;
        if (dd > (denom * qq))
            break;

        dd += qq;
        denom += 1;
        selected_indices.push_back(std::make_pair(j, kk));
    }

    return std::make_tuple(dd / denom, denom, selected_indices);
}

std::tuple<float, int, index_vec> subproblem(
    torch::TensorAccessor<float, 1> eta,
    torch::TensorAccessor<float, 2> q,
    torch::TensorAccessor<long, 2> q_srt,
    const std::vector<long>& color,
    const int c
){
    const auto n_nodes = q.size(0);
    const auto n_samples = q.size(1);

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
            dd_init += eta[j];
            n_rows += 1;
        }
    }

    assert(n_rows >= 1);

    auto dd = dd_init;
    int denom = n_rows;

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

            auto kk = q_srt[j][k];
            auto qq = q[j][kk];

            if (qq > qmax) {
                qmax = qq;
                jjmax = jj;
                jmax = j;
                kkmax = kk;
            }
        }

        if (jjmax < 0)
            break;
        if (dd_init > (n_rows * qmax)) // is this check redundant?
            break;
        if (dd > (denom * qmax))
            break;

        dd += qmax;
        denom += 1;
        ks[jjmax] += 1; // advance jth row head
        selected_indices.push_back(std::make_pair(jmax, kkmax));
    }
    return std::make_tuple(dd / denom, denom, selected_indices);
}


std::tuple<torch::Tensor, std::vector<long>, std::vector<int>, std::vector<index_vec>>
compute_d_fast(torch::Tensor q, torch::Tensor eta) {
    q = q.t();
    const int n_nodes = q.size(0);

    auto parent = [&n_nodes](int t) {return (t - 1) / 2;} ;

    // initialize each node to its own color
    auto color = std::vector<long>(n_nodes);
    std::iota(color.begin(), color.end(), 0);
    auto color_to_ix = std::vector<index_vec>(n_nodes);

    // precompute descending sort of all the qs
    auto q_srt = q.argsort(1, true);

    auto q_acc = q.accessor<float, 2>();
    auto q_srt_acc = q_srt.accessor<long, 2>();
    auto eta_acc = eta.accessor<float, 1>();

    // output
    auto d = eta.clone();
    auto d_acc = d.accessor<float, 1>();

    auto denoms = std::vector<int>(n_nodes);

    // solve all nested qps to initialize
    for(int j = 0; j < n_nodes; ++j) {
        std::tie(d_acc[j], denoms[j], color_to_ix[j]) = subproblem_init(eta_acc, q_acc, q_srt_acc, j);
    }

    int n_iter = 0;

    while (true) {
        n_iter += 1;

        // find most violating edge
        int max_viol_ix = -1;
        float max_viol_d = -std::numeric_limits<float>::infinity();
        for (int t = 1; t < n_nodes; ++t) {
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
        int pc = color[p];
        int tc = color[t];
        //std::cout << "joining " << t << " into " << p << std::endl;

        color_to_ix[tc] = {};
        denoms[tc] = 0;

        for (auto& c : color) {
            if (c == tc)
                c = pc;
        }

        // update d[color == pc];
        float dcol;
        std::tie(dcol, denoms[pc], color_to_ix[pc]) = subproblem(eta_acc, q_acc, q_srt_acc, color, pc);
        for(int j = 0; j < n_nodes; ++j) {
            if (color[j] == pc)
                d_acc[j] = dcol;
        }
    }
    return std::make_tuple(d, color, denoms, color_to_ix);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_d_fast", &compute_d_fast, "todo");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
