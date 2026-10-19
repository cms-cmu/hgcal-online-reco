// Graphs the per-event wafer count and k (number of trainable clusters),
// and reports the smallest MAX_WAFERS / MAX_K that still cover a chosen
// fraction of events, so you can pick the smallest model input sizes.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include "TCanvas.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TStyle.h"
#include "ROOT/RDataFrame.hxx"

namespace {

// Smallest value v such that at least `frac` of the (sorted) entries are <= v
int Quantile(const std::vector<int> &sorted, double frac) {
    size_t idx = static_cast<size_t>(std::ceil(frac * sorted.size()));
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return sorted[idx - 1];
}

// Fraction of events with value <= cap, for cap = 0..maxVal
TGraph *CoverageGraph(const std::vector<int> &vals, int maxVal, const char *name) {
    std::vector<double> counts(maxVal + 1, 0.0);
    for (int v : vals) counts[v] += 1.0;
    auto *g = new TGraph(maxVal + 1);
    g->SetName(name);
    double running = 0.0;
    for (int c = 0; c <= maxVal; ++c) {
        running += counts[c];
        g->SetPoint(c, c, running / vals.size());
    }
    return g;
}

} // namespace

void graph_k_wafers() {
    ROOT::EnableImplicitMT();

    ROOT::RDataFrame df("Events", "/home/export/dylankan/hgcal_online_reco/data/*.root");

    auto dfCounts = df.Define("wafer_count", "static_cast<int>(L1THGCAL_wafer_x.size())")
                       .Define("k_count", "static_cast<int>(Sum(MergedSimCluster_isTrainable, 0))")
                       .Filter("k_count >= 1");

    // Single event loop: pull both columns into memory (paired by event)
    auto wafersLazy = dfCounts.Take<int>("wafer_count");
    auto kLazy = dfCounts.Take<int>("k_count");
    std::vector<int> wafers = wafersLazy.GetValue();
    std::vector<int> ks = kLazy.GetValue();

    const size_t n = wafers.size();
    if (n == 0) {
        std::cout << "No events passed the filter." << std::endl;
        return;
    }

    const int maxW = *std::max_element(wafers.begin(), wafers.end());
    const int minW = *std::min_element(wafers.begin(), wafers.end());
    const int maxK = *std::max_element(ks.begin(), ks.end());
    const int minK = *std::min_element(ks.begin(), ks.end());

    std::vector<int> wSorted(wafers), kSorted(ks);
    std::sort(wSorted.begin(), wSorted.end());
    std::sort(kSorted.begin(), kSorted.end());

    // ---- Report: smallest cap needed per coverage target ----
    const std::vector<double> targets = {0.90, 0.95, 0.99, 0.999, 0.9999, 1.0};
    std::cout << "=====================================================" << std::endl;
    std::cout << "Events (k>=1): " << n << std::endl;
    std::cout << "wafers: min " << minW << ", max " << maxW << std::endl;
    std::cout << "k     : min " << minK << ", max " << maxK << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << std::setw(10) << "coverage" << std::setw(12) << "MAX_WAFERS"
              << std::setw(8) << "MAX_K" << std::setw(22) << "events kept (both)" << std::endl;
    for (double t : targets) {
        int capW = Quantile(wSorted, t);
        int capK = Quantile(kSorted, t);
        size_t both = 0;
        for (size_t i = 0; i < n; ++i)
            if (wafers[i] <= capW && ks[i] <= capK) ++both;
        std::cout << std::setw(9) << std::fixed << std::setprecision(2) << 100.0 * t << "%"
                  << std::setw(12) << capW << std::setw(8) << capK
                  << std::setw(14) << std::setprecision(3) << 100.0 * both / n << "%" << std::endl;
    }
    std::cout << "=====================================================" << std::endl;

    const int capW99 = Quantile(wSorted, 0.99);
    const int capK99 = Quantile(kSorted, 0.99);

    // ---- Plots ----
    gStyle->SetOptStat("emr");
    TCanvas c("c1", "k and wafers", 1400, 1000);
    c.Divide(2, 2);

    // 1) wafers histogram (bin width 1 wafer, or coarser if range is large)
    int wBins = std::min(maxW + 1, 200);
    auto *hW = new TH1I("hW", "Wafers per event;Number of wafers;Events", wBins, -0.5, maxW + 0.5);
    for (int v : wafers) hW->Fill(v);
    hW->SetLineWidth(2);
    c.cd(1);
    gPad->SetLogy();
    hW->Draw();

    // 2) k histogram (bin width 1)
    auto *hK = new TH1I("hK", "k per event;k (trainable clusters);Events", maxK + 1, -0.5, maxK + 0.5);
    for (int v : ks) hK->Fill(v);
    hK->SetLineWidth(2);
    hK->SetLineColor(kRed + 1);
    c.cd(2);
    gPad->SetLogy();
    hK->Draw();

    // 3) 2D: wafers vs k
    auto *h2 = new TH2D("h2", "Wafers vs k;k (trainable clusters);Number of wafers",
                        maxK + 1, -0.5, maxK + 0.5, wBins, -0.5, maxW + 0.5);
    for (size_t i = 0; i < n; ++i) h2->Fill(ks[i], wafers[i]);
    c.cd(3);
    gPad->SetLogz();
    gStyle->SetOptStat(0);
    h2->Draw("COLZ");

    // 4) coverage curves: fraction of events a given cap would keep
    c.cd(4);
    gPad->SetGrid();
    auto *gW = CoverageGraph(wafers, maxW, "gW");
    auto *gK = CoverageGraph(ks, maxK, "gK");
    gW->SetTitle("Fraction of events covered by a cap;Cap (MAX_WAFERS or MAX_K);Fraction of events");
    gW->SetLineColor(kBlue + 1);
    gW->SetLineWidth(2);
    gK->SetLineColor(kRed + 1);
    gK->SetLineWidth(2);
    gW->GetYaxis()->SetRangeUser(0.0, 1.02);
    gW->Draw("AL");
    gK->Draw("L SAME");
    auto *leg = new TLegend(0.55, 0.15, 0.88, 0.32);
    leg->AddEntry(gW, "wafers", "l");
    leg->AddEntry(gK, "k", "l");
    leg->Draw();

    c.SaveAs("k_wafers_summary.png");
    std::cout << "Saved k_wafers_summary.png" << std::endl;
    std::cout << "99% caps -> MAX_WAFERS=" << capW99 << ", MAX_K=" << capK99 << std::endl;
}