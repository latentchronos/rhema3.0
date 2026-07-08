# ProPresenter Integration

This document defines the integration boundary for ProPresenter support.

## Goal

Rhema should be able to send the current verse reference and text to a ProPresenter-controlled display workflow without replacing existing NDI, display-window, or OBS outputs.

## Recommended First Adapter

Use a dedicated `ProPresenterOutput` adapter beside the existing broadcast output paths.

The adapter should accept the same normalized payload used by OBS:

- reference, such as `John 3:16`;
- verse text;
- translation;
- output target, such as main or alternate;
- theme metadata when ProPresenter needs it.

The adapter should then translate that payload into ProPresenter-specific actions.

## Operator Settings

Add these settings before enabling live push:

- host, default `127.0.0.1`;
- port, from the local ProPresenter setup;
- authentication token or password, stored locally;
- target playlist, presentation, macro, stage screen, or prop field;
- dry-run/test button.

## Safety Requirements

- Never log authentication tokens.
- Keep ProPresenter output disabled until connection testing succeeds.
- Preserve manual operator control: detected verses should not surprise-push to ProPresenter unless auto mode is explicitly enabled.
- Add a visible connection state and last-error message.

## Implementation Steps

1. Confirm the ProPresenter API version and endpoint contract for the installed ProPresenter version.
2. Add a Rust command module for connection testing and payload push.
3. Add frontend settings for host, port, token, and target.
4. Add a `propresenter` broadcast output option.
5. Add integration tests around payload normalization and error handling.

The current phase adds the documented adapter contract and security baseline. Live ProPresenter API calls should be implemented only after endpoint verification against the target ProPresenter version.
