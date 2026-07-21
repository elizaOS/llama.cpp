export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["favicon.svg","loading.html"]),
	mimeTypes: {".svg":"image/svg+xml",".html":"text/html"},
	_: {
		client: {start:"_app/immutable/bundle.DT4SV6KR.js",imports:["_app/immutable/bundle.DT4SV6KR.js"],stylesheets:["_app/immutable/assets/bundle.D6fOGM0E.css"],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js')),
			__memo(() => import('./nodes/3.js')),
			__memo(() => import('./nodes/4.js')),
			__memo(() => import('./nodes/5.js')),
			__memo(() => import('./nodes/6.js')),
			__memo(() => import('./nodes/7.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/(chat)",
				pattern: /^\/?$/,
				params: [],
				page: { layouts: [0,2,], errors: [1,,], leaf: 4 },
				endpoint: null
			},
			{
				id: "/(chat)/chat/[id]",
				pattern: /^\/chat\/([^/]+?)\/?$/,
				params: [{"name":"id","optional":false,"rest":false,"chained":false}],
				page: { layouts: [0,2,], errors: [1,,], leaf: 5 },
				endpoint: null
			},
			{
				id: "/mcp-servers",
				pattern: /^\/mcp-servers\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 6 },
				endpoint: null
			},
			{
				id: "/settings/[[section]]",
				pattern: /^\/settings(?:\/([^/]+))?\/?$/,
				params: [{"name":"section","optional":true,"rest":false,"chained":true}],
				page: { layouts: [0,3,], errors: [1,,], leaf: 7 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
