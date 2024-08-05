function [log_old, log_new] = ensure_continuity(log_old, log_new)
%-------------------------------------------------------------------------
    if norm(log_old - log_new) > pi
        log_new = (norm(log_new) - 2*pi*round(norm(log_old-log_new)/(2*pi))) * ...
            log_new/norm(log_new);
    end
    if norm(log_old - log_new) > pi
        log_new = (2*pi*round(norm(log_old-log_new)/(2*pi)) - norm(log_new)) * ...
            log_new/norm(log_new);
    end
    log_old = log_new;
end