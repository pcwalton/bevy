//! BSN asset loading: parse a `.bsn` file containing named asset definitions
//! and insert them into Bevy asset stores via reflection.

use bevy_asset::{AssetPath, AssetServer};
use bevy_ecs::{prelude::*, reflect::AppTypeRegistry};
use bevy_reflect::{prelude::ReflectDefault, ReflectMut, TypeRegistry};

use crate::dynamic_bsn::{BsnAst, BsnExpr, BsnNameStore, BsnPatch, BsnPatches};
use crate::dynamic_bsn_grammar::TopLevelPatchesParser;
use crate::dynamic_bsn_lexer::Lexer;

/// Parse a BSN file containing named asset definitions and insert them into
/// the corresponding `Assets<T>` stores via reflection.
///
/// Returns a list of `(name, UntypedHandle)` pairs for the created assets.
pub fn load_bsn_assets(
    world: &mut World,
    bsn_text: &str,
) -> Result<Vec<(String, bevy_asset::UntypedHandle)>, String> {
    let mut parse_world = World::new();
    parse_world.init_resource::<BsnNameStore>();
    let ast = core::cell::RefCell::new(BsnAst(parse_world));

    let lexer = Lexer::new(bsn_text);
    let patches_id = TopLevelPatchesParser::new()
        .parse(&ast, lexer)
        .map_err(|e| format!("BSN asset parse error: {e:?}"))?;

    let bsn_ast = ast.into_inner();
    let root_entries = unwrap_roots(&bsn_ast, patches_id)?;

    let registry = world.resource::<AppTypeRegistry>().clone();
    let reg = registry.read();
    let asset_server = world.get_resource::<AssetServer>().cloned();

    let mut results = Vec::new();

    for entry_id in root_entries {
        let Some(patches) = bsn_ast.0.get::<BsnPatches>(entry_id) else { continue };

        let name = patches.0.iter().find_map(|&pid| match bsn_ast.0.get::<BsnPatch>(pid)? {
            BsnPatch::Name(n, _) => Some(n.clone()),
            _ => None,
        });
        let Some(name) = name else { continue };

        for &patch_id in &patches.0 {
            let Some(patch) = bsn_ast.0.get::<BsnPatch>(patch_id) else { continue };
            let handle = match patch {
                BsnPatch::Struct(bsn_struct) => {
                    create_asset(world, &bsn_struct.0.as_path(), Some(&bsn_struct.1), &bsn_ast, &reg, asset_server.as_ref())
                }
                BsnPatch::Var(var) => {
                    create_asset(world, &var.0.as_path(), None, &bsn_ast, &reg, asset_server.as_ref())
                }
                _ => None,
            };
            if let Some(handle) = handle {
                results.push((name.clone(), handle));
            }
        }
    }

    Ok(results)
}

fn unwrap_roots(ast: &BsnAst, patches_id: Entity) -> Result<Vec<Entity>, String> {
    let patches = ast.0.get::<BsnPatches>(patches_id)
        .ok_or_else(|| "No top-level patches found".to_string())?;
    if patches.0.len() == 1 {
        if let Some(BsnPatch::Relation(relation)) = ast.0.get::<BsnPatch>(patches.0[0]) {
            return Ok(relation.1.clone());
        }
    }
    Ok(vec![patches_id])
}

fn create_asset(
    world: &mut World,
    type_path: &str,
    fields: Option<&[crate::dynamic_bsn::BsnField]>,
    ast: &BsnAst,
    registry: &TypeRegistry,
    asset_server: Option<&AssetServer>,
) -> Option<bevy_asset::UntypedHandle> {
    let registration = registry.get_with_type_path(type_path)?;
    let reflect_default = registration.data::<ReflectDefault>()?;
    let mut value = reflect_default.default();

    if let (Some(fields), Ok(struct_info)) = (fields, registration.type_info().as_struct()) {
        if let ReflectMut::Struct(s) = value.reflect_mut() {
            for field in fields {
                let Some(fi) = struct_info.field(&field.0) else { continue };
                let Some(expr) = ast.0.get::<BsnExpr>(field.1) else { continue };
                apply_expr(s, &field.0, expr, fi.ty().id(), registry, asset_server, ast);
            }
        }
    }

    let reflect_asset = registration.data::<bevy_asset::ReflectAsset>()?;
    Some(reflect_asset.add(world, value.as_partial_reflect()))
}

fn apply_expr(
    target: &mut dyn bevy_reflect::structs::Struct,
    field_name: &str,
    expr: &BsnExpr,
    expected: core::any::TypeId,
    registry: &TypeRegistry,
    asset_server: Option<&AssetServer>,
    ast: &BsnAst,
) {
    let Some(field) = target.field_mut(field_name) else { return };

    match expr {
        BsnExpr::FloatLit(f) => {
            if expected == core::any::TypeId::of::<f32>() { field.apply(&(*f as f32)); }
            else if expected == core::any::TypeId::of::<f64>() { field.apply(f); }
        }
        BsnExpr::IntLit(i) => {
            macro_rules! try_int {
                ($($t:ty),*) => { $(if expected == core::any::TypeId::of::<$t>() { field.apply(&(*i as $t)); return; })* };
            }
            try_int!(i8, u8, i16, u16, i32, u32, i64, u64, usize, isize);
        }
        BsnExpr::BoolLit(b) => { field.apply(b); }
        BsnExpr::StringLit(s) => {
            // For Handle<T> fields, skip — asset loading from paths is handled
            // by the scene's HandleTemplate resolution, not the catalog.
            if let Some(concrete) = field.try_as_reflect() {
                let tid = concrete.reflect_type_info().type_id();
                if registry.get_type_data::<bevy_asset::ReflectHandle>(tid).is_some() {
                    return; // Handle fields resolved during scene spawn
                }
            }
            if expected == core::any::TypeId::of::<String>() { field.apply(s); }
        }
        BsnExpr::Struct(bsn_struct) => {
            let type_path = bsn_struct.0.as_path();
            let Some(reg) = registry.get_with_type_path(&type_path) else { return };
            let Ok(si) = reg.type_info().as_struct() else { return };
            let ReflectMut::Struct(s) = field.reflect_mut() else { return };
            for f in &bsn_struct.1 {
                let Some(fi) = si.field(&f.0) else { continue };
                let Some(e) = ast.0.get::<BsnExpr>(f.1) else { continue };
                apply_expr(s, &f.0, e, fi.ty().id(), registry, asset_server, ast);
            }
        }
        _ => {}
    }
}
