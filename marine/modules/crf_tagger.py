# Copyright 2019 Allen Institute for AI
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Conditional random field
"""

import math
from logging import getLogger

import torch


logger = getLogger(__name__)

VITERBI_DECODING = tuple[list[int], float]  # a list of tags, and a viterbi score


def logsumexp(
    tensor: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.
    This function is typically used for summing log probabilities.
    # Parameters
    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    tag_observations: list[int] | None = None,
    allowed_start_transitions: torch.Tensor | None = None,
    allowed_end_transitions: torch.Tensor | None = None,
    top_k: int | None = None,
):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    # Parameters
    tag_sequence : `torch.Tensor`, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : `torch.Tensor`, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : `Optional[List[int]]`, optional, (default = `None`)
        A list of length `sequence_length` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    top_k : `int`, optional, (default = `None`)
        Optional integer specifying how many of the top paths to return. For top_k>=1, returns
        a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
        tuple with just the top path and its score (not in lists, for backwards compatibility).
    # Returns
    viterbi_path : `List[int]`
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : `torch.Tensor`
        The score of the viterbi path.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(
            f"top_k must be either None or an integer >=1. Instead received {top_k}"
        )

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = (
        allowed_end_transitions is not None or allowed_start_transitions is not None
    )

    if has_start_end_restrictions:
        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.

        allowed_start_transitions = torch.cat(
            [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        allowed_end_transitions = torch.cat(
            [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
        )

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ValueError(
                "Observations were provided, but they were not the same length "
                f"as the sequence. Found sequence of length: {sequence_length} and evidence: {tag_observations}"
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning(
                    "The pairwise potential between tags you have passed as "
                    "observations is extremely unlikely. Double check your evidence "
                    "or transition potentials!"
                )
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()

        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]

        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]

    return viterbi_paths, viterbi_scores


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.
    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf
    # Parameters
    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_tags: int,
        constraints: list[tuple[int, int]] | None = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1)
        # and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 1.0)
        else:
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(
        self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood
        This is the sum of the likelihoods across all possible state sequences.
        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor of
                unnormalized log-probabilities
            transitions (torch.Tensor): a (batch_size, num_tags, num_tags) tensor of
                transition scores
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags
        Returns:
            torch.Tensor: (batch_size,) denominator term $Z(x)$, per example,
                for the log-likelihood
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag")
            # so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag)
            # so we broadcast along the instance axis.
            transition_scores = transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == True)
            # we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(
        self,
        logits: torch.Tensor,
        transitions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor
                of unnormalized log-probabilities
            transitions (torch.Tensor): a (batch_size, num_tags, num_tags) tensor
                of transition scores
            tags (torch.Tensor): output tag sequences (batch_size, sequence_length) $y$
                for each input sequence
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags
        Returns:
            torch.Tensor: numerator term for the log-likelihood,
                which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag)
            transition_score = transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state.
        # To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(
            1, last_tags.view(-1, 1)
        )  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Computes the log likelihood for the given batch of input sequences $(x,y)$
        Args:
            inputs (torch.Tensor): (batch_size, sequence_length, num_tags) tensor
                of logits for the inputs $x$
            tags (torch.Tensor): (batch_size, sequence_length) tensor of tags $y$
            mask (torch.BoolTensor, optional): (batch_size, sequence_length) tensor
                of masking flags. Defaults to None.
        Returns:
            torch.Tensor: (batch_size,) log likelihoods $log P(y|x)$ for each input
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool)

        log_denominator = self._input_likelihood(inputs, self.transitions, mask)
        log_numerator = self._joint_likelihood(inputs, self.transitions, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
        self,
        logits: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        top_k: int | None = None,
    ) -> list[VITERBI_DECODING] | list[list[VITERBI_DECODING]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)
        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.full(
            (num_tags + 2, num_tags + 2), -10000.0, device=logits.device
        )

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask[
            :num_tags, :num_tags
        ] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                self.start_transitions.detach()
                * self._constraint_mask[start_tag, :num_tags].data
                + -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                self.end_transitions.detach()
                * self._constraint_mask[:num_tags, end_tag].data
                + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - self._constraint_mask[:num_tags, end_tag].detach()
            )

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.empty(
            max_seq_length + 2, num_tags + 2, device=logits.device
        )

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1 : (sequence_length + 1), :num_tags] = masked_prediction
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to `viterbi_decode`.
            viterbi_paths, viterbi_scores = viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths
