
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/private';
 * 
 * console.log(ENVIRONMENT); // => "production"
 * console.log(PUBLIC_BASE_URL); // => throws error during build
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/private' {
	export const _ZO_DOCTOR: string;
	export const MANPATH: string;
	export const VSCODE_CRASH_REPORTER_PROCESS_TYPE: string;
	export const NODE: string;
	export const INIT_CWD: string;
	export const SHELL: string;
	export const TERM: string;
	export const VSCODE_PROCESS_TITLE: string;
	export const MAKEFLAGS: string;
	export const TMPDIR: string;
	export const npm_config_global_prefix: string;
	export const CURSOR_WORKSPACE_LABEL: string;
	export const MallocNanoZone: string;
	export const SDKROOT: string;
	export const COLOR: string;
	export const NO_COLOR: string;
	export const npm_config_noproxy: string;
	export const npm_config_local_prefix: string;
	export const CURSOR_LAYOUT: string;
	export const CURSOR_CONVERSATION_ID: string;
	export const USER: string;
	export const CGO_LDFLAGS: string;
	export const COMMAND_MODE: string;
	export const npm_config_globalconfig: string;
	export const LLAMA_CPP_ROOT: string;
	export const SSH_AUTH_SOCK: string;
	export const __CF_USER_TEXT_ENCODING: string;
	export const npm_execpath: string;
	export const MAKELEVEL: string;
	export const MFLAGS: string;
	export const PATH: string;
	export const npm_package_json: string;
	export const npm_config_engine_strict: string;
	export const _: string;
	export const LaunchInstanceID: string;
	export const npm_config_userconfig: string;
	export const npm_config_init_module: string;
	export const __CFBundleIdentifier: string;
	export const npm_command: string;
	export const PWD: string;
	export const VSCODE_HANDLES_UNCAUGHT_ERRORS: string;
	export const npm_lifecycle_event: string;
	export const EDITOR: string;
	export const VSCODE_ESM_ENTRYPOINT: string;
	export const npm_package_name: string;
	export const CURSOR_AGENT: string;
	export const LANG: string;
	export const npm_config_npm_version: string;
	export const XPC_FLAGS: string;
	export const CURSOR_EXTENSION_HOST_ROLE: string;
	export const FORCE_COLOR: string;
	export const MACH_PORT_RENDEZVOUS_PEER_VALDATION: string;
	export const npm_config_node_gyp: string;
	export const npm_package_version: string;
	export const CGO_ENABLED: string;
	export const XPC_SERVICE_NAME: string;
	export const CXX: string;
	export const HOME: string;
	export const SHLVL: string;
	export const VSCODE_NLS_CONFIG: string;
	export const npm_config_cache: string;
	export const LOGNAME: string;
	export const npm_lifecycle_script: string;
	export const CURSOR_RIPGREP_PATH: string;
	export const VSCODE_CODE_CACHE_PATH: string;
	export const VSCODE_IPC_HOOK: string;
	export const PKG_CONFIG_PATH: string;
	export const BUN_INSTALL: string;
	export const npm_config_user_agent: string;
	export const VSCODE_PID: string;
	export const DISPLAY: string;
	export const CC: string;
	export const AGENT_TRANSCRIPTS: string;
	export const OSLogRateLimit: string;
	export const VSCODE_CWD: string;
	export const SECURITYSESSIONID: string;
	export const npm_node_execpath: string;
	export const npm_config_prefix: string;
	export const NODE_ENV: string;
}

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/public';
 * 
 * console.log(ENVIRONMENT); // => throws error during build
 * console.log(PUBLIC_BASE_URL); // => "http://site.com"
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * 
 * console.log(env.ENVIRONMENT); // => "production"
 * console.log(env.PUBLIC_BASE_URL); // => undefined
 * ```
 */
declare module '$env/dynamic/private' {
	export const env: {
		_ZO_DOCTOR: string;
		MANPATH: string;
		VSCODE_CRASH_REPORTER_PROCESS_TYPE: string;
		NODE: string;
		INIT_CWD: string;
		SHELL: string;
		TERM: string;
		VSCODE_PROCESS_TITLE: string;
		MAKEFLAGS: string;
		TMPDIR: string;
		npm_config_global_prefix: string;
		CURSOR_WORKSPACE_LABEL: string;
		MallocNanoZone: string;
		SDKROOT: string;
		COLOR: string;
		NO_COLOR: string;
		npm_config_noproxy: string;
		npm_config_local_prefix: string;
		CURSOR_LAYOUT: string;
		CURSOR_CONVERSATION_ID: string;
		USER: string;
		CGO_LDFLAGS: string;
		COMMAND_MODE: string;
		npm_config_globalconfig: string;
		LLAMA_CPP_ROOT: string;
		SSH_AUTH_SOCK: string;
		__CF_USER_TEXT_ENCODING: string;
		npm_execpath: string;
		MAKELEVEL: string;
		MFLAGS: string;
		PATH: string;
		npm_package_json: string;
		npm_config_engine_strict: string;
		_: string;
		LaunchInstanceID: string;
		npm_config_userconfig: string;
		npm_config_init_module: string;
		__CFBundleIdentifier: string;
		npm_command: string;
		PWD: string;
		VSCODE_HANDLES_UNCAUGHT_ERRORS: string;
		npm_lifecycle_event: string;
		EDITOR: string;
		VSCODE_ESM_ENTRYPOINT: string;
		npm_package_name: string;
		CURSOR_AGENT: string;
		LANG: string;
		npm_config_npm_version: string;
		XPC_FLAGS: string;
		CURSOR_EXTENSION_HOST_ROLE: string;
		FORCE_COLOR: string;
		MACH_PORT_RENDEZVOUS_PEER_VALDATION: string;
		npm_config_node_gyp: string;
		npm_package_version: string;
		CGO_ENABLED: string;
		XPC_SERVICE_NAME: string;
		CXX: string;
		HOME: string;
		SHLVL: string;
		VSCODE_NLS_CONFIG: string;
		npm_config_cache: string;
		LOGNAME: string;
		npm_lifecycle_script: string;
		CURSOR_RIPGREP_PATH: string;
		VSCODE_CODE_CACHE_PATH: string;
		VSCODE_IPC_HOOK: string;
		PKG_CONFIG_PATH: string;
		BUN_INSTALL: string;
		npm_config_user_agent: string;
		VSCODE_PID: string;
		DISPLAY: string;
		CC: string;
		AGENT_TRANSCRIPTS: string;
		OSLogRateLimit: string;
		VSCODE_CWD: string;
		SECURITYSESSIONID: string;
		npm_node_execpath: string;
		npm_config_prefix: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://example.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.ENVIRONMENT); // => undefined, not public
 * console.log(env.PUBLIC_BASE_URL); // => "http://example.com"
 * ```
 * 
 * ```
 * 
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
