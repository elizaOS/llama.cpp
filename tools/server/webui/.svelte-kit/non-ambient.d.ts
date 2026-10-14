
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	type MatcherParam<M> = M extends (param : string) => param is (infer U extends string) ? U : string;

	export interface AppTypes {
		RouteId(): "/(chat)" | "/" | "/(chat)/chat" | "/(chat)/chat/[id]" | "/mcp-servers" | "/settings" | "/settings/[[section]]";
		RouteParams(): {
			"/(chat)/chat/[id]": { id: string };
			"/settings/[[section]]": { section?: string }
		};
		LayoutParams(): {
			"/(chat)": { id?: string };
			"/": { id?: string; section?: string };
			"/(chat)/chat": { id?: string };
			"/(chat)/chat/[id]": { id: string };
			"/mcp-servers": Record<string, never>;
			"/settings": { section?: string };
			"/settings/[[section]]": { section?: string }
		};
		Pathname(): "/" | `/chat/${string}` & {} | "/mcp-servers" | `/settings${string}` & {};
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): "/favicon.svg" | "/loading.html" | string & {};
	}
}