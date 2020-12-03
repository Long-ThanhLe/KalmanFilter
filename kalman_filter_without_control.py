import numpy as np

def kalman_predict(
    pre_state_estimate,
    observation,
    pre_cov,
    transition_matrix,
    observation_matrix,
    process_noise_cov,
    observation_noise_cov
):
    prior_error_cov = np.dot(
        np.dot(transition_matrix, pre_cov), 
        transition_matrix.T
    ) + process_noise_cov
    prior_state_estimate = np.dot(
        transition_matrix,
        pre_state_estimate
    )
    
    kalman_gain = np.dot(
        np.dot(prior_error_cov, observation_matrix.T),
        np.linalg.inv(
            np.dot(
                np.dot(
                    observation_matrix,
                    prior_error_cov
                ),
                observation_matrix.T
            ) +  observation_noise_cov
        )
    )
    state_estimate = prior_state_estimate + np.dot(
        kalman_gain,
        observation - np.dot(observation_matrix, prior_state_estimate)
    )
    error_cov = np.dot(
        np.identity(kalman_gain.shape[0]) - np.dot(kalman_gain, observation_matrix),
        prior_error_cov
    )
    
    kalman_value = {
        "gain" : kalman_gain,
        "prior_cov" : prior_error_cov,
        "prior_estimate" : prior_state_estimate,
        "cov" : error_cov,
        "estimate" : state_estimate
    }
    return kalman_value