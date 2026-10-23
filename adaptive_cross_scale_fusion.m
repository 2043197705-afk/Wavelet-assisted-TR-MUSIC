function [fused_curve, info] = adaptive_cross_scale_fusion(subband_response, opts)
%ADAPTIVE_CROSS_SCALE_FUSION Adaptive fusion of multiscale response curves.
% Input:
%   subband_response : Nsubband-by-Ndepth matrix; each row is one subband
%                      response curve sampled on the same axial-depth grid.
%   opts             : optional structure for algorithm parameters.
% Output:
%   fused_curve      : max-normalized fused response curve.
%   info             : intermediate values useful for analysis and plotting.

if nargin < 2 || isempty(opts)
    opts = struct();
end

opts = set_default(opts, 'subband_smooth_win', 3);
opts = set_default(opts, 'use_subband_smoothing', false);
opts = set_default(opts, 'use_final_smoothing', false);
opts = set_default(opts, 'global_noise_floor_abs', 0.015);
opts = set_default(opts, 'global_noise_floor_ratio', 0.25);
opts = set_default(opts, 'global_noise_reference_quantile', 90);
opts = set_default(opts, 'confidence_tau_z', 6.0);
opts = set_default(opts, 'confidence_temperature_z', 1.0);
opts = set_default(opts, 'support_confidence_threshold', 0.60);
opts = set_default(opts, 'multiplicative_gain_eta', 1.50);
opts = set_default(opts, 'support_penalty_kappa', 0.90);
opts = set_default(opts, 'top2_second_evidence_weight', 0.65);
opts = set_default(opts, 'global_reliability_floor', 0.10);
opts = set_default(opts, 'global_reliability_power', 0.50);
opts = set_default(opts, 'top2_confidence_gain', 0.65);
opts = set_default(opts, 'mix_lambda_max', 0.80);

validateattributes(subband_response, {'numeric'}, {'2d','nonempty'}, mfilename, 'subband_response', 1);
if size(subband_response, 2) < 2
    error('subband_response 至少需要包含两个轴向采样点。');
end
if opts.confidence_temperature_z <= 0
    error('opts.confidence_temperature_z 必须大于 0。');
end
if opts.subband_smooth_win < 1
    error('opts.subband_smooth_win 必须大于或等于 1。');
end

P_raw = abs(double(subband_response));
P_raw(~isfinite(P_raw)) = 0;
[num_subbands, num_depths] = size(P_raw);

if opts.use_subband_smoothing
    P_work = smoothdata(P_raw, 2, 'movmedian', opts.subband_smooth_win);
else
    P_work = P_raw;
end

% Normalize each subband independently.
normalized_subband = zeros(size(P_work));
for s = 1:num_subbands
    row = P_work(s, :);
    peak_value = max(row);
    if isfinite(peak_value) && peak_value > 0
        normalized_subband(s, :) = row ./ peak_value;
    end
end

% Estimate a global reliability weight for each subband.
global_reliability_raw = zeros(num_subbands, 1);
for s = 1:num_subbands
    t = log1p(P_raw(s, :));
    med_t = median(t);
    mad_t = median(abs(t - med_t)) + eps;
    score = (max(t) - med_t) ./ mad_t;
    if ~isfinite(score) || score < 0
        score = 0;
    end
    global_reliability_raw(s) = score;
end

reliability_compressed = (global_reliability_raw + opts.global_reliability_floor) .^ opts.global_reliability_power;
global_reliability = reliability_compressed ./ (sum(reliability_compressed) + eps);

% Estimate the noise baseline and confidence at every depth sample.
global_baseline = zeros(num_subbands, 1);
global_noise_mad = zeros(num_subbands, 1);
global_noise_quantile_floor = zeros(num_subbands, 1);
global_noise_sigma = zeros(num_subbands, 1);
global_zscore = zeros(size(normalized_subband));
confidence = zeros(size(normalized_subband));

for s = 1:num_subbands
    R = normalized_subband(s, :);
    baseline = median(R);
    mad_sigma = 1.4826 * median(abs(R - baseline));
    qref = prctile(R, opts.global_noise_reference_quantile);
    quantile_floor = opts.global_noise_floor_ratio * max(qref - baseline, 0);
    sigma = max([mad_sigma, opts.global_noise_floor_abs, quantile_floor]);
    Z = (R - baseline) ./ (sigma + eps);
    C = 1 ./ (1 + exp(-(Z - opts.confidence_tau_z) ./ opts.confidence_temperature_z));

    global_baseline(s) = baseline;
    global_noise_mad(s) = mad_sigma;
    global_noise_quantile_floor(s) = quantile_floor;
    global_noise_sigma(s) = sigma;
    global_zscore(s, :) = Z;
    confidence(s, :) = C;
end

% Apply a confidence-dependent gain, penalizing locations supported by many subbands.
effective_support = sum(confidence, 1);
multiplicative_gain = zeros(size(normalized_subband));
shape_preserving_evidence = zeros(size(normalized_subband));
for s = 1:num_subbands
    other_support = max(effective_support - confidence(s, :), 0);
    gain = 1 + opts.multiplicative_gain_eta .* confidence(s, :) ./ ...
        (1 + opts.support_penalty_kappa .* other_support);
    multiplicative_gain(s, :) = gain;
    shape_preserving_evidence(s, :) = normalized_subband(s, :) .* gain;
end

% Select the two strongest confidence-weighted subband responses at each depth.
ranking_score = confidence .* normalized_subband;
[~, ranking_index] = sort(ranking_score, 1, 'descend');
cols = 1:num_depths;
top1_subband_index = ranking_index(1, :);
lin_top1 = sub2ind(size(normalized_subband), top1_subband_index, cols);
top1_response = normalized_subband(lin_top1);
top1_confidence = confidence(lin_top1);
top1_gain = multiplicative_gain(lin_top1);
top1_evidence = shape_preserving_evidence(lin_top1);

if num_subbands >= 2
    top2_subband_index = ranking_index(2, :);
    lin_top2 = sub2ind(size(normalized_subband), top2_subband_index, cols);
    top2_response = normalized_subband(lin_top2);
    top2_confidence = confidence(lin_top2);
    top2_gain = multiplicative_gain(lin_top2);
    top2_evidence = shape_preserving_evidence(lin_top2);
else
    top2_subband_index = zeros(1, num_depths);
    top2_response = zeros(1, num_depths);
    top2_confidence = zeros(1, num_depths);
    top2_gain = ones(1, num_depths);
    top2_evidence = zeros(1, num_depths);
end

top2_denominator = top1_confidence + opts.top2_second_evidence_weight .* top2_confidence;
top2_convex_curve = (top1_confidence .* top1_evidence + ...
    opts.top2_second_evidence_weight .* top2_confidence .* top2_evidence) ./ ...
    (top2_denominator + eps);
top2_convex_curve(top2_denominator < 1e-8) = 0;

% Form a stable consensus using the global reliability weights.
consensus_curve = sum(bsxfun(@times, global_reliability, normalized_subband), 1);

% Adaptively mix the consensus curve and the top-two evidence curve.
sorted_confidence = sort(confidence, 1, 'descend');
c1_mix = sorted_confidence(1, :);
if num_subbands >= 2
    c2_mix = sorted_confidence(2, :);
else
    c2_mix = zeros(1, num_depths);
end
adaptive_mix = opts.mix_lambda_max .* ...
    (1 - (1 - c1_mix) .* (1 - opts.top2_confidence_gain .* c2_mix));
adaptive_mix = min(max(adaptive_mix, 0), opts.mix_lambda_max);

final_curve_raw = (1 - adaptive_mix) .* consensus_curve + adaptive_mix .* top2_convex_curve;
if opts.use_final_smoothing
    final_curve_raw = smoothdata(final_curve_raw, 'movmean', 3);
end
fused_curve = normalize_max_only(final_curve_raw);
support_count = sum(confidence >= opts.support_confidence_threshold, 1);

info.P_subband_raw = P_raw;
info.P_work = P_work;
info.normalized_subband = normalized_subband;
info.global_reliability_raw = global_reliability_raw;
info.global_reliability = global_reliability;
info.global_baseline = global_baseline;
info.global_noise_mad = global_noise_mad;
info.global_noise_quantile_floor = global_noise_quantile_floor;
info.global_noise_sigma = global_noise_sigma;
info.global_zscore = global_zscore;
info.confidence = confidence;
info.effective_support = effective_support;
info.multiplicative_gain = multiplicative_gain;
info.shape_preserving_evidence = shape_preserving_evidence;
info.ranking_score = ranking_score;
info.top1_subband_index = top1_subband_index;
info.top2_subband_index = top2_subband_index;
info.top1_response = top1_response;
info.top2_response = top2_response;
info.top1_confidence = top1_confidence;
info.top2_confidence = top2_confidence;
info.top1_gain = top1_gain;
info.top2_gain = top2_gain;
info.top1_evidence = top1_evidence;
info.top2_evidence = top2_evidence;
info.top2_convex_curve = top2_convex_curve;
info.consensus_curve = consensus_curve;
info.support_count = support_count;
info.adaptive_mix = adaptive_mix;
info.final_curve_raw = final_curve_raw;
info.final_curve_normalized = fused_curve;
info.options = opts;
end

function opts = set_default(opts, field_name, default_value)
if ~isfield(opts, field_name) || isempty(opts.(field_name))
    opts.(field_name) = default_value;
end
end

function y = normalize_max_only(x)
y = double(x(:)).';
y(~isfinite(y)) = 0;
peak_value = max(y);
if peak_value > 0
    y = y ./ peak_value;
end
end
