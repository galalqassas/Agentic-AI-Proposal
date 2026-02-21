/**
 * Proposal Agent — Custom Frontend Injector
 * 1. Stacks the Copy button with the Edit button in a vertical column.
 * 2. Replaces standard Step avatars with Lottie animations.
 */
(function() {
  // 1. Inject dotLottie Player script if missing
  if (!document.querySelector('script[src*="dotlottie-player"]')) {
    const script = document.createElement('script');
    script.type = 'module';
    script.src = 'https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs';
    document.head.appendChild(script);
  }

  const ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  const OK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';

  const AVATARS = {
    'Planner': 'planner.lottie',
    'Researcher': 'researcher.lottie',
    'Writer': 'writer.lottie',
    'Evaluator': 'eval.lottie'
  };

  const inject = () => {
    // Inject Copy Buttons
    document.querySelectorAll('[data-step-type="user_message"]').forEach(m => {
      const edit = m.querySelector('.edit-message');
      if (!edit || edit.parentNode.classList.contains('cl-actions')) return;

      const wrap = document.createElement('div');
      wrap.className = 'cl-actions flex flex-col items-center gap-0.5 ml-auto invisible group-hover:visible';
      
      edit.classList.remove('ml-auto', 'invisible', 'group-hover:visible');
      edit.parentNode.insertBefore(wrap, edit);
      wrap.append(edit);

      const btn = document.createElement('button');
      btn.className = 'h-8 w-8 flex items-center justify-center hover:bg-accent rounded-md transition-colors text-muted-foreground';
      btn.innerHTML = ICON;
      btn.title = 'Copy message';
      btn.onclick = () => navigator.clipboard.writeText(m.querySelector('.message-content').innerText).then(() => {
        btn.innerHTML = OK;
        btn.classList.add('text-green-400');
        setTimeout(() => { btn.innerHTML = ICON; btn.classList.remove('text-green-400'); }, 1500);
      });
      wrap.append(btn);
    });

    // Inject Lottie Avatars
    document.querySelectorAll('button[id^="step-"]').forEach(btn => {
      const text = btn.innerText || btn.textContent;
      if (!text) return;

      Object.entries(AVATARS).forEach(([agent, lottieFile]) => {
        if (text.includes(agent)) {
          // Find the main chat row for this step
          const row = btn.closest('.ai-message');
          if (!row) return;

          // The first child of the .ai-message row is the avatar wrapper span
          const avatarWrapper = row.firstElementChild;
          if (!avatarWrapper || avatarWrapper.tagName.toLowerCase() !== 'span') return;

          // Prevent duplication within the same row
          if (row.hasAttribute('data-lottie-injected')) return;
          row.setAttribute('data-lottie-injected', 'true');

          // Clean up old players in case of hot-reloads
          Array.from(avatarWrapper.querySelectorAll('dotlottie-player')).forEach(p => p.remove());

          // Hide the original Chainlit placeholder (grey circle)
          Array.from(avatarWrapper.children).forEach(child => {
            child.style.setProperty('display', 'none', 'important');
          });
          
          // Inject our dotLottie player explicitly into the avatar column
          const player = document.createElement('dotlottie-player');
          player.setAttribute('src', `/public/avatars/${lottieFile}`);
          player.setAttribute('autoplay', '');
          player.setAttribute('loop', '');
          player.style.width = '24px';  // Matches Chainlit default sizing
          player.style.height = '24px';
          player.style.display = 'block';
          
          // Align it nicely
          avatarWrapper.style.display = 'flex';
          avatarWrapper.style.alignItems = 'center';
          avatarWrapper.style.justifyContent = 'center';

          avatarWrapper.appendChild(player);
        }
      });
    });
  };

  new MutationObserver(() => requestAnimationFrame(inject)).observe(document.body, {childList: true, subtree: true});
  inject();
})();
