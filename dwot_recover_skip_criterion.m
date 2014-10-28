function skip_criteria = dwot_recover_skip_criterion(one_char_skip_criteria)

skip_criteria= cell(1, numel(one_char_skip_criteria));

char_idx = 1;
for skip_char = one_char_skip_criteria
    curr_skip_criterion = '';
    switch skip_char
        case 'e'
            curr_skip_criterion = 'empty';
        case 'd'
            curr_skip_criterion = 'difficult';
        case 'o'
            curr_skip_criterion = 'occluded';
        case 't'
            curr_skip_criterion = 'truncated';
    end
    skip_criteria{char_idx} = curr_skip_criterion;
    char_idx = char_idx + 1;
end