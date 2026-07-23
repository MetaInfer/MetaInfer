#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace metainfer::reference {

enum class NumericWeightFormat {
    kF16,
    kQ8_0,
};

struct NumericFeatures {
    NumericWeightFormat weight_format = NumericWeightFormat::kF16;
    bool paged_kv = false;
    bool continuous_batching = false;
    bool tensor_parallel = false;
};

inline std::vector<std::string> required_numeric_case_ids(
    const NumericFeatures& features) {
    std::vector<std::string> required{
        "cast_fp32_to_fp16",
        "rms_norm",
        "per_head_rms_norm",
        "rope_neox",
        "kv_write",
        "prefill_gqa",
        "swiglu",
        "greedy",
    };
    if (features.weight_format == NumericWeightFormat::kF16) {
        required.push_back("f16_linear");
    } else {
        required.push_back("dequant_q8_0");
        required.push_back("q8_embedding");
        required.push_back("q8_linear");
    }
    if (features.paged_kv) {
        required.push_back("paged_attention");
    }
    if (features.continuous_batching) {
        required.push_back("packed_sequence_isolation");
    }
    if (features.tensor_parallel) {
        required.push_back("tp_collective");
        required.push_back("tp_sharded_linear");
    }
    if (features.paged_kv || features.continuous_batching) {
        required.push_back("kv_capacity_contract");
    }
    return required;
}

struct NumericCaseResult {
    std::string id;
    bool passed = false;
    std::string detail;
};

struct NumericRunReport {
    bool passed = false;
    std::vector<NumericCaseResult> cases;
};

class NumericHarness {
public:
    using CaseFunction = std::function<NumericCaseResult()>;

    bool add(std::string id, CaseFunction function) {
        if (id.empty() || !function) {
            return false;
        }
        return cases_.emplace(std::move(id), std::move(function)).second;
    }

    NumericRunReport run_required(const NumericFeatures& features) const {
        NumericRunReport report;
        report.passed = true;
        for (const std::string& id : required_numeric_case_ids(features)) {
            const auto found = cases_.find(id);
            if (found == cases_.end()) {
                report.cases.push_back(
                    NumericCaseResult{id, false, "required case is not registered"});
                report.passed = false;
                continue;
            }
            NumericCaseResult result = found->second();
            if (result.id.empty()) {
                result.id = id;
            }
            if (result.id != id) {
                result.passed = false;
                result.detail = "case returned a different id";
                result.id = id;
            }
            report.passed = report.passed && result.passed;
            report.cases.push_back(std::move(result));
        }
        return report;
    }

private:
    std::map<std::string, CaseFunction> cases_;
};

inline std::string numeric_json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    static const char* digits = "0123456789abcdef";
                    output << "\\u00" << digits[ch >> 4U] << digits[ch & 0x0fU];
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

inline std::string numeric_report_json(const NumericRunReport& report) {
    std::ostringstream output;
    output << "{\"passed\":" << (report.passed ? "true" : "false")
           << ",\"cases\":[";
    for (std::size_t index = 0; index < report.cases.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const NumericCaseResult& result = report.cases[index];
        output << "{\"id\":\"" << numeric_json_escape(result.id)
               << "\",\"passed\":" << (result.passed ? "true" : "false")
               << ",\"detail\":\"" << numeric_json_escape(result.detail)
               << "\"}";
    }
    output << "]}";
    return output.str();
}

}  // namespace metainfer::reference
