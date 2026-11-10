//! What the viewer asked the worker for, what it shows, and the copies that show frames again.

use std::sync::Arc;

use crate::document::NodeId;
use crate::image::{ImageInfo, Window};
use crate::viewer_cache::{ViewerCache, ViewerCopy};
use crate::worker::Job;

/// What the viewer shows, as [`App::displayed`](super::App::displayed) describes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Displayed {
    /// The node the image is of, which the viewer may no longer mark.
    pub node: NodeId,
    /// The frame the image is of, which may not be the session's while that one is evaluated.
    pub frame: f64,
    /// Whether the frame failed, and shows no image.
    pub failed: bool,
    /// How many times the viewer cache's copy shown was halved, or `None` if the image shown was
    /// evaluated.
    pub copy_halvings: Option<u32>,
}

#[derive(Debug, Default)]
pub(super) struct Viewing {
    pub(super) cache: ViewerCache,
    pub(super) asked: Asked,
    pub(super) shown: Shown,
}

/// What the viewer last wanted, and where it comes from.
#[derive(Debug, Default)]
pub(super) enum Asked {
    #[default]
    Nothing,
    /// The worker was asked for it.
    Evaluating(Job),
    /// The worker is paused while the viewer shows the cache's copy of it.
    ServedFromCache(Job),
}

impl Asked {
    pub(super) fn job(&self) -> Option<&Job> {
        match self {
            Asked::Nothing => None,
            Asked::Evaluating(job) | Asked::ServedFromCache(job) => Some(job),
        }
    }
}

/// What the viewer shows.
#[derive(Debug, Default)]
pub(super) enum Shown {
    #[default]
    Nothing,
    Image {
        job: Job,
        info: ImageInfo,
    },
    /// A copy from the viewer cache.
    Copy {
        job: Job,
        copy: Arc<ViewerCopy>,
    },
    /// An empty frame, never an older image.
    Failed {
        job: Job,
        error: String,
    },
}

impl Shown {
    /// The job whose result is shown.
    pub(super) fn job(&self) -> Option<&Job> {
        match self {
            Shown::Nothing => None,
            Shown::Image { job, .. } | Shown::Copy { job, .. } | Shown::Failed { job, .. } => {
                Some(job)
            }
        }
    }

    /// The image the viewer frames: none for an empty display window, which has no size to fit.
    pub(super) fn framed(&self) -> Option<&ImageInfo> {
        let info = match self {
            Shown::Image { info, .. } => info,
            Shown::Copy { copy, .. } => &copy.info,
            Shown::Nothing | Shown::Failed { .. } => return None,
        };
        (!info.display_window.is_empty()).then_some(info)
    }

    pub(super) fn display_window(&self) -> Option<Window> {
        self.framed().map(|info| info.display_window)
    }
}

impl Viewing {
    /// Whether the viewer shows the result of what it last asked for: an image, a copy or a
    /// failure.
    pub(super) fn shows_asked(&self) -> bool {
        match (self.asked.job(), self.shown.job()) {
            (Some(asked), Some(shown)) => shown.key == asked.key,
            _ => false,
        }
    }

    /// Whether the viewer shows the result of what it last asked for, as the frame's.
    pub(super) fn shows_asked_at(&self, frame: f64) -> bool {
        self.shows_asked() && (self.shown.job()).is_some_and(|shown| shown.frame == frame)
    }

    pub(super) fn displayed(&self) -> Option<Displayed> {
        let job = self.shown.job()?;
        Some(Displayed {
            node: job.node,
            frame: job.frame,
            failed: matches!(self.shown, Shown::Failed { .. }),
            copy_halvings: match &self.shown {
                Shown::Copy { copy, .. } => Some(copy.halvings()),
                Shown::Nothing | Shown::Image { .. } | Shown::Failed { .. } => None,
            },
        })
    }
}
