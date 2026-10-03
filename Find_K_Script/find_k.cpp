#include <iostream>
#include "ROOT/RDataFrame.hxx"

void find_k() {
    ROOT::EnableImplicitMT();

    // Init RDataFrame for all data events in folder
    ROOT::RDataFrame df("Events", "/home/export/dylankan/hgcal_online_reco/data/*.root");

    // Define new columns for the wafer count and k count per event
    auto dfCounts = df.Define("wafer_count", "L1THGCAL_wafer_x.size()")
                       .Define("k_count", "Sum(MergedSimCluster_isTrainable, 0)");

    // Register lazy actions to find the maximum values of these new columns
    auto maxWafers = dfCounts.Max("wafer_count");
    auto maxK = dfCounts.Max("k_count");

    std::cout << "=====================================" << std::endl;
    std::cout << "Extracted Maximums:" << std::endl;
    std::cout << "max_wafers : " << maxWafers.GetValue() << std::endl;
    std::cout << "max_k      : " << maxK.GetValue() << std::endl;
    std::cout << "=====================================" << std::endl;
}