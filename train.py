# Pseudocode for InfoMamba.


def infomamba_block(tokens, recurrent_state, parameters):
    # Adaptive concept assignment and active concept selection.
    projected = project(tokens, parameters.assignment_projection)
    assignment = softmax(similarity(projected, parameters.prototypes) / parameters.temperature)
    active = select_active_concepts(assignment, parameters.activity_threshold)
    assignment = renormalize(assignment[:, active])

    # Write, mix, and read the compact concept state.
    concepts = assignment.transpose(-1, -2) @ projected
    concepts = concepts + multi_head_attention(layer_norm(concepts))
    concept_signal = gelu(layer_norm(assignment @ concepts))

    # Dual-path injection into recurrence and output representations.
    next_state = recurrent_update(tokens, recurrent_state, parameters) + state_projection(concept_signal)
    output = recurrent_readout(next_state, parameters) + output_projection(concept_signal)
    return output, next_state, concepts


def train_step(batch, model, optimizer, beta, gamma):
    inputs, targets = batch
    outputs, recurrent_features, concept_features = model.forward_with_states(inputs, infomamba_block)

    task_loss = cross_entropy(outputs, targets)
    global_loss = cross_entropy(model.concept_head(pool(concept_features)), targets)
    redundancy_loss = squared_frobenius_norm(cross_correlation(pool(concept_features), pool(recurrent_features)))
    loss = task_loss + beta * global_loss + gamma * redundancy_loss

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss


def train(train_loader, model, optimizer, scheduler, beta, gamma):
    for batch in train_loader:
        train_step(batch, model, optimizer, beta, gamma)
        scheduler.step()
